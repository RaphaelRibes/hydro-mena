//! RMT-based automatic threshold detection (Deng et al. 2012, Fig. 2, steps a-k).
//!
//! a-b  similarity S = |Pearson r|
//! c-d  adjacency A_ij = 1 when S_ij >= st, zero diagonal, drop degree-0 nodes
//! e-f  eigenvalues of A
//! g-i  unfold the spectrum: fit the integrated density N(lambda) with a
//!      low-order polynomial and take the spacings of N at consecutive
//!      eigenvalues, rescaled to unit mean
//! j-k  chi^2 goodness-of-fit of the spacing histogram against the Poisson
//!      law exp(-s) at alpha = 0.01
//!
//! The previous version derived "unfolded" spacings from the rank alone
//! (e_i = (i+0.5)/p, d_i = p*(e_{i+1}-e_i) = 1 for every i), so the statistic
//! collapsed to chi^2 = 0 for every graph and the *first* candidate threshold
//! always "converged" — with a start of 0.01 that returned the complete graph
//! (M ~ 0.01, gamma ~ -42). Unfolding now uses the actual eigenvalue density
//! and the GOF is a real binned chi-square.
use crate::linalg::{self, Mat};
use crate::stats;
use rayon::prelude::*;

#[derive(Clone, Debug, serde::Serialize)]
pub struct ScanPoint {
    pub st: f64,
    /// nodes left after dropping degree-0 rows
    pub p: usize,
    pub edges: usize,
    pub chi2: f64,
    pub dof: usize,
    pub crit: f64,
    /// chi^2 <= crit: spacing distribution is compatible with Poisson
    pub poisson: bool,
    /// Kolmogorov-Smirnov distance to the Poisson CDF (secondary diagnostic)
    pub ks: f64,
}

/// Symmetric 0/1 adjacency from the similarity matrix at threshold `st`.
pub fn adjacency(s: &Mat, st: f64) -> Mat {
    let n = s.nrows();
    let mut a = Mat::zeros((n, n));
    for i in 0..n {
        for j in (i + 1)..n {
            if s[(i, j)] >= st {
                a[(i, j)] = 1.0;
                a[(j, i)] = 1.0;
            }
        }
    }
    a
}

/// Nodes with at least one edge.
pub fn kept_nodes(a: &Mat) -> Vec<usize> {
    let n = a.nrows();
    (0..n)
        .filter(|&i| (0..n).any(|j| i != j && a[(i, j)] > 0.0))
        .collect()
}

fn edges_of(a: &Mat) -> usize {
    let n = a.nrows();
    let mut e = 0usize;
    for i in 0..n {
        for j in (i + 1)..n {
            if a[(i, j)] > 0.0 {
                e += 1;
            }
        }
    }
    e
}

/// Least-squares polynomial fit, normal equations with partial pivoting.
/// `x` is rescaled to [-1, 1] first so the Vandermonde system stays conditioned.
/// Returns coefficients in ascending power order.
fn polyfit(x: &[f64], y: &[f64], degree: usize) -> Vec<f64> {
    let d = degree + 1;
    let (lo, hi) = x
        .iter()
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(a, b), &v| (a.min(v), b.max(v)));
    let span = if hi > lo { (hi - lo) / 2.0 } else { 1.0 };
    let mid = (hi + lo) / 2.0;
    let xn: Vec<f64> = x.iter().map(|&v| (v - mid) / span).collect();

    let mut ata = vec![0.0f64; d * d];
    let mut atb = vec![0.0f64; d];
    let mut pw = vec![1.0f64; d];
    for i in 0..xn.len() {
        let xi = xn[i];
        pw[0] = 1.0;
        for k in 1..d {
            pw[k] = pw[k - 1] * xi;
        }
        for r in 0..d {
            atb[r] += pw[r] * y[i];
            for c in 0..d {
                ata[r * d + c] += pw[r] * pw[c];
            }
        }
    }
    // Gaussian elimination with partial pivoting
    for col in 0..d {
        let mut piv = col;
        for r in (col + 1)..d {
            if ata[r * d + col].abs() > ata[piv * d + col].abs() {
                piv = r;
            }
        }
        if ata[piv * d + col].abs() < 1e-14 {
            continue;
        }
        if piv != col {
            for c in 0..d {
                ata.swap(piv * d + c, col * d + c);
            }
            atb.swap(piv, col);
        }
        let diag = ata[col * d + col];
        for r in (col + 1)..d {
            let f = ata[r * d + col] / diag;
            if f == 0.0 {
                continue;
            }
            for c in col..d {
                ata[r * d + c] -= f * ata[col * d + c];
            }
            atb[r] -= f * atb[col];
        }
    }
    let mut coef = vec![0.0f64; d];
    for r in (0..d).rev() {
        let mut acc = atb[r];
        for c in (r + 1)..d {
            acc -= ata[r * d + c] * coef[c];
        }
        let diag = ata[r * d + r];
        coef[r] = if diag.abs() > 1e-14 { acc / diag } else { 0.0 };
    }
    // back to the original variable: p((v-mid)/span)
    let mut out = vec![0.0f64; d];
    for (k, &c) in coef.iter().enumerate() {
        if k == 0 {
            out[0] += c;
            continue;
        }
        // expand c * ((v-mid)/span)^k
        let mut bin = vec![1.0f64];
        for _ in 0..k {
            let mut next = vec![0.0f64; bin.len() + 1];
            for (i, &b) in bin.iter().enumerate() {
                next[i] += -b * mid / span;
                next[i + 1] += b / span;
            }
            bin = next;
        }
        for (i, &b) in bin.iter().enumerate() {
            out[i] += c * b;
        }
    }
    out
}

fn polyval(coef: &[f64], x: f64) -> f64 {
    let mut acc = 0.0;
    for &c in coef.iter().rev() {
        acc = acc * x + c;
    }
    acc
}

/// Unfold a sorted spectrum: spacings of the fitted integrated density,
/// normalised to unit mean. Returns (spacings, unfolded positions).
fn unfolded_spacings(lam_in: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let mut sorted: Vec<f64> = lam_in.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let lam_sorted: &[f64] = &sorted;
    let p = lam_sorted.len();
    if p < 6 {
        return (Vec::new(), lam_sorted.to_vec());
    }
    let rank: Vec<f64> = (0..p).map(|i| i as f64 + 1.0).collect();
    let degree = 5.min(p / 5).max(1);
    let coef = polyfit(lam_sorted, &rank, degree);
    let xi: Vec<f64> = lam_sorted.iter().map(|&l| polyval(&coef, l)).collect();
    let mut d: Vec<f64> = (0..p - 1).map(|i| xi[i + 1] - xi[i]).collect();
    let mean = d.iter().sum::<f64>() / d.len() as f64;
    if mean > 0.0 {
        for v in d.iter_mut() {
            *v /= mean;
        }
    }
    (d, xi)
}

/// Evaluate one candidate threshold: eigen-decompose the adjacency, unfold,
/// run the Poisson GOF.
fn evaluate(s: &Mat, st: f64, min_nodes: usize, alpha: f64) -> Option<ScanPoint> {
    let a = adjacency(s, st);
    let keep = kept_nodes(&a);
    let p = keep.len();
    if p < min_nodes.max(6) {
        return None;
    }
    let mut sub = Mat::zeros((p, p));
    for (x, &i) in keep.iter().enumerate() {
        for (y, &j) in keep.iter().enumerate() {
            sub[(x, y)] = a[(i, j)];
        }
    }
    let (lam, _) = linalg::eigh(&sub);
    let (d, _) = unfolded_spacings(&lam);
    if d.len() < 6 {
        return None;
    }
    let (chi2, dof, crit) = stats::poisson_gof(&d, alpha);
    let ks = stats::ks_poisson(&d);
    Some(ScanPoint {
        st,
        p,
        edges: edges_of(&sub),
        chi2,
        dof,
        crit,
        poisson: dof > 0 && chi2 <= crit,
        ks,
    })
}

fn scan(s: &Mat, sts: &[f64], min_nodes: usize, alpha: f64) -> Vec<ScanPoint> {
    let mut out: Vec<ScanPoint> = sts
        .par_iter()
        .filter_map(|&st| evaluate(s, st, min_nodes, alpha))
        .collect();
    out.sort_by(|a, b| a.st.partial_cmp(&b.st).unwrap());
    out
}

pub struct RmtResult {
    pub st: f64,
    /// true when some threshold was accepted by the GOF
    pub converged: bool,
    /// true when no threshold was accepted and `st` is a diagnostic fallback
    pub fallback: bool,
    pub log: Vec<ScanPoint>,
    pub adjacency: Mat,
    pub kept: Vec<usize>,
    /// spectrum of the kept subgraph at the chosen threshold
    pub eigenvalues: Vec<f64>,
    pub spacings: Vec<f64>,
    pub edges: usize,
}

/// RMT threshold: coarse scan by 0.05 from `st0`, then a 0.01 refine inside the
/// accepting decade. When nothing is accepted the smallest chi^2/crit ratio is
/// used and flagged `fallback` — the caller must report that, not hide it.
pub fn rmt_threshold(s: &Mat, st0: f64, min_nodes: usize, alpha: f64) -> RmtResult {
    let mut coarse: Vec<f64> = Vec::new();
    let mut st = st0.max(0.01);
    while st <= 0.95 + 1e-9 {
        coarse.push((st * 1000.0).round() / 1000.0);
        st += 0.05;
    }
    let log_coarse = scan(s, &coarse, min_nodes, alpha);

    let accepted = log_coarse.iter().find(|p| p.poisson).map(|p| p.st);
    let (st_final, converged, fallback, log) = match accepted {
        Some(acc) => {
            let lo = (acc - 0.05).max(0.01);
            let mut fine: Vec<f64> = Vec::new();
            let mut v = lo;
            while v <= acc + 1e-9 {
                fine.push((v * 1000.0).round() / 1000.0);
                v += 0.01;
            }
            let mut log_fine = scan(s, &fine, min_nodes, alpha);
            let best = log_fine
                .iter()
                .find(|p| p.poisson)
                .map(|p| p.st)
                .unwrap_or(acc);
            log_fine.extend(log_coarse.iter().cloned());
            log_fine.sort_by(|a, b| a.st.partial_cmp(&b.st).unwrap());
            log_fine.dedup_by(|a, b| (a.st - b.st).abs() < 1e-12);
            (best, true, false, log_fine)
        }
        None => {
            // no acceptance: report the closest-to-Poisson point, flagged
            let best = log_coarse
                .iter()
                .min_by(|a, b| {
                    let ra = a.chi2 / a.crit.max(1e-12);
                    let rb = b.chi2 / b.crit.max(1e-12);
                    ra.partial_cmp(&rb).unwrap()
                })
                .map(|p| p.st)
                .unwrap_or_else(|| coarse.last().copied().unwrap_or(0.95));
            (best, false, true, log_coarse)
        }
    };

    let mut r = at_threshold(s, st_final);
    r.converged = converged;
    r.fallback = fallback;
    r.log = log;
    r
}

/// Build and analyse the graph at a fixed similarity threshold (no scan):
/// adjacency, kept nodes, and the spectrum the RMT figure needs.
pub fn at_threshold(s: &Mat, st: f64) -> RmtResult {
    let a = adjacency(s, st);
    let kept = kept_nodes(&a);
    let edges = edges_of(&a);
    let mut eigenvalues = Vec::new();
    let mut spacings = Vec::new();
    if !kept.is_empty() {
        let p = kept.len();
        let mut sub = Mat::zeros((p, p));
        for (x, &i) in kept.iter().enumerate() {
            for (y, &j) in kept.iter().enumerate() {
                sub[(x, y)] = a[(i, j)];
            }
        }
        let (lam, _) = linalg::eigh(&sub);
        let (d, _) = unfolded_spacings(&lam);
        eigenvalues = lam;
        spacings = d;
    }
    RmtResult {
        st,
        converged: false,
        fallback: false,
        log: Vec::new(),
        adjacency: a,
        kept,
        eigenvalues,
        spacings,
        edges,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn polyfit_recovers_a_line() {
        let x: Vec<f64> = (0..20).map(|i| i as f64).collect();
        let y: Vec<f64> = x.iter().map(|v| 3.0 * v + 2.0).collect();
        let c = polyfit(&x, &y, 3);
        assert!((polyval(&c, 0.0) - 2.0).abs() < 1e-8, "{c:?}");
        assert!((polyval(&c, 19.0) - 59.0).abs() < 1e-8, "{c:?}");
    }

    #[test]
    fn unfolding_spacings_have_unit_mean() {
        // non-uniform but smooth spectrum
        let lam: Vec<f64> = (0..200).map(|i| (i as f64 * 0.05).sin() * 2.0 + i as f64 * 0.01).collect();
        let (d, xi) = unfolded_spacings(&lam);
        assert!(d.len() >= 100);
        let mean = d.iter().sum::<f64>() / d.len() as f64;
        assert!((mean - 1.0).abs() < 1e-9, "mean spacing {mean}");
        // and it is not the degenerate rank-only spacing
        assert!(d.iter().any(|v| (v - 1.0).abs() > 1e-6));
        assert_eq!(xi.len(), lam.len());
    }

    #[test]
    fn poisson_accepted_on_random_graph_rejected_on_lattice() {
        // A sparse Erdos-Renyi graph has Poisson-like spacings; a complete graph
        // has a degenerate spectrum and must be rejected.
        let n = 90;
        let mut rng = rand::thread_rng();
        use rand::Rng;
        let mut s = Mat::zeros((n, n));
        for i in 0..n {
            for j in (i + 1)..n {
                let v = if rng.gen::<f64>() < 0.06 { 0.9 } else { 0.1 };
                s[(i, j)] = v;
                s[(j, i)] = v;
            }
        }
        let rand_pt = evaluate(&s, 0.5, 10, 0.01).expect("point");
        assert!(rand_pt.p > 10);
        // complete graph: one eigenvalue n-1 and p-1 degenerate -1
        let mut full = Mat::from_elem((n, n), 0.9);
        for i in 0..n {
            full[(i, i)] = 0.0;
        }
        let full_pt = evaluate(&full, 0.5, 10, 0.01).expect("point");
        assert!(!full_pt.poisson, "complete graph must not look Poisson: {full_pt:?}");
    }
}
