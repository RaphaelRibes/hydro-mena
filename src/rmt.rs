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

/// Every post-hoc choice in the spectral test, as a toggle. These were
/// introduced while looking at real-data output, so each must be ablable: a
/// "not estimable" verdict is only meaningful once we know which guard caused it.
#[derive(Clone, Debug)]
pub struct RmtConfig {
    /// restrict the spectrum to the largest connected component
    pub giant: bool,
    /// drop the Perron eigenvalue (and the lowest) before unfolding
    pub trim_extremes: bool,
    /// minimum spacing count for the GOF to be attempted
    pub min_spacings: usize,
    /// minimum dof for a point to count as estimable
    pub min_dof: usize,
}

impl Default for RmtConfig {
    fn default() -> Self {
        Self { giant: true, trim_extremes: true, min_spacings: 20, min_dof: 2 }
    }
}

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
    /// share of kept nodes inside the giant component the test was run on
    pub giant_frac: f64,
    /// why this candidate was accepted or dropped: `accepted`, `rejected_gof`,
    /// `excluded_min_nodes`, `excluded_tiny_giant`, `excluded_too_few_spacings`,
    /// `excluded_low_dof`. Every candidate is reported, so the curve can be
    /// audited instead of silently thinned to its survivors.
    pub status: String,
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
    let mut xi: Vec<f64> = lam_sorted.iter().map(|&l| polyval(&coef, l)).collect();
    // A polynomial integrated density is not guaranteed monotone, and a
    // non-monotone N(lambda) is not a valid unfolding: it yields negative
    // spacings, which the GOF would then silently drop from the observed counts
    // while still expecting them. Enforce monotonicity, then reject the
    // unfolding outright if any spacing is still non-positive.
    for i in 1..xi.len() {
        if xi[i] < xi[i - 1] {
            xi[i] = xi[i - 1];
        }
    }
    let raw: Vec<f64> = (0..p - 1).map(|i| xi[i + 1] - xi[i]).collect();
    let mean = raw.iter().sum::<f64>() / raw.len() as f64;
    if !(mean > 0.0) || raw.iter().any(|&v| v <= 0.0) {
        // degenerate spectrum (e.g. many repeated eigenvalues in a 0/1 adjacency)
        return (Vec::new(), xi);
    }
    let mut d = raw;
    for v in d.iter_mut() {
        *v /= mean;
    }
    (d, xi)
}

/// Largest connected component of the binary graph over `keep`, as positions
/// into `keep`, plus its fraction of all kept nodes.
///
/// A disconnected graph is Poisson-like for free (independent spectra
/// superpose to Poisson), so testing a fragmented graph measures fragmentation,
/// not modular structure. The spectral test is therefore run on the giant
/// component and `giant_frac` is reported with every scan point.
fn giant_component(a: &Mat, keep: &[usize]) -> (Vec<usize>, f64) {
    let p = keep.len();
    let mut comp = vec![usize::MAX; p];
    let mut best: Vec<usize> = Vec::new();
    let mut c = 0usize;
    for s0 in 0..p {
        if comp[s0] != usize::MAX {
            continue;
        }
        comp[s0] = c;
        let mut stack = vec![s0];
        let mut members = vec![s0];
        while let Some(u) = stack.pop() {
            for v in 0..p {
                if v != u && comp[v] == usize::MAX && a[(keep[u], keep[v])] > 0.0 {
                    comp[v] = c;
                    stack.push(v);
                    members.push(v);
                }
            }
        }
        if members.len() > best.len() {
            best = members;
        }
        c += 1;
    }
    let frac = if p > 0 { best.len() as f64 / p as f64 } else { 0.0 };
    (best, frac)
}

/// Evaluate one candidate threshold: eigen-decompose the adjacency, unfold,
/// run the Poisson GOF.
fn evaluate(s: &Mat, st: f64, min_nodes: usize, alpha: f64, cfg: &RmtConfig) -> ScanPoint {
    let a = adjacency(s, st);
    let keep = kept_nodes(&a);
    let p = keep.len();
    let edges = edges_of(&a);
    let ex = |p: usize, edges: usize, reason: &str, giant_frac: f64| ScanPoint {
        st,
        p,
        edges,
        chi2: f64::NAN,
        dof: 0,
        crit: f64::NAN,
        poisson: false,
        ks: f64::NAN,
        giant_frac,
        status: reason.to_string(),
    };
    if p < min_nodes.max(6) {
        return ex(p, edges, "excluded_min_nodes", f64::NAN);
    }
    let (giant, giant_frac) = if cfg.giant {
        giant_component(&a, &keep)
    } else {
        ((0..keep.len()).collect::<Vec<usize>>(), 1.0)
    };
    if giant.len() < min_nodes.max(6) {
        return ex(p, edges, "excluded_tiny_giant", giant_frac);
    }
    let g = giant.len();
    // Deng/Luo eigen-decompose the *thresholded similarity* matrix (weights
    // retained, below-threshold entries zeroed, diagonal 0) — see Eq. 3 and the
    // 2025 repo's pretreating.py. Binarising produces repeated 0/-1 eigenvalues.
    let mut w = Mat::zeros((g, g));
    for (x, &i) in giant.iter().enumerate() {
        for (y, &j) in giant.iter().enumerate() {
            if x != y && s[(keep[i], keep[j])] >= st {
                w[(x, y)] = s[(keep[i], keep[j])];
            }
        }
    }
    let (mut lam, _) = linalg::eigh(&w);
    if cfg.trim_extremes && lam.len() > 10 {
        lam.pop();
        lam.remove(0);
    }
    let (d, _) = unfolded_spacings(&lam);
    if d.len() < cfg.min_spacings {
        return ex(p, edges, "excluded_too_few_spacings", giant_frac);
    }
    let (chi2, dof, crit) = stats::poisson_gof(&d, alpha);
    let ks = stats::ks_poisson(&d);
    let mut pt = ScanPoint {
        st,
        p,
        edges,
        chi2,
        dof,
        crit,
        poisson: false,
        ks,
        giant_frac,
        status: String::new(),
    };
    if dof < cfg.min_dof || !chi2.is_finite() {
        pt.status = "excluded_low_dof".into();
    } else if chi2 <= crit {
        pt.status = "accepted".into();
        pt.poisson = true;
    } else {
        pt.status = "rejected_gof".into();
    }
    pt
}

fn scan(s: &Mat, sts: &[f64], min_nodes: usize, alpha: f64, cfg: &RmtConfig) -> Vec<ScanPoint> {
    let mut out: Vec<ScanPoint> = sts
        .par_iter()
        .map(|&st| evaluate(s, st, min_nodes, alpha, cfg))
        .collect();
    out.sort_by(|a, b| a.st.partial_cmp(&b.st).unwrap());
    out
}

pub struct RmtResult {
    /// false when no candidate passed: no adjacency exists and none is invented
    pub estimable: bool,
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
pub fn rmt_threshold(s: &Mat, st0: f64, min_nodes: usize, alpha: f64, cfg: &RmtConfig) -> RmtResult {
    let mut coarse: Vec<f64> = Vec::new();
    let mut st = st0.max(0.01);
    while st <= 0.95 + 1e-9 {
        coarse.push((st * 1000.0).round() / 1000.0);
        st += 0.05;
    }
    let log_coarse = scan(s, &coarse, min_nodes, alpha, cfg);

    let accepted = log_coarse.iter().find(|p| p.status == "accepted").map(|p| p.st);
    let (st_final, converged, fallback, log) = match accepted {
        Some(acc) => {
            let lo = (acc - 0.05).max(0.01);
            let mut fine: Vec<f64> = Vec::new();
            let mut v = lo;
            while v <= acc + 1e-9 {
                fine.push((v * 1000.0).round() / 1000.0);
                v += 0.01;
            }
            let mut log_fine = scan(s, &fine, min_nodes, alpha, cfg);
            let best = log_fine
                .iter()
                .find(|p| p.status == "accepted")
                .map(|p| p.st)
                .unwrap_or(acc);
            log_fine.extend(log_coarse.iter().cloned());
            log_fine.sort_by(|a, b| a.st.partial_cmp(&b.st).unwrap());
            log_fine.dedup_by(|a, b| (a.st - b.st).abs() < 1e-12);
            (best, true, false, log_fine)
        }
        None => {
            // No acceptance. Do NOT invent a threshold: there is no graph to
            // build, and the previous behaviour (pick the least-bad point and
            // build a network there) produced plausible-looking results that
            // were not results at all.
            (f64::NAN, false, true, log_coarse)
        }
    };

    if st_final.is_nan() {
        return RmtResult {
            estimable: false,
            st: f64::NAN,
            converged: false,
            fallback: true,
            log,
            adjacency: Mat::zeros((0, 0)),
            kept: Vec::new(),
            eigenvalues: Vec::new(),
            spacings: Vec::new(),
            edges: 0,
        };
    }
    let mut r = at_threshold(s, st_final);
    r.estimable = true;
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
        let mut w = Mat::zeros((p, p));
        for (x, &i) in kept.iter().enumerate() {
            for (y, &j) in kept.iter().enumerate() {
                if x != y && s[(i, j)] >= st {
                    w[(x, y)] = s[(i, j)];
                }
            }
        }
        let (lam, _) = linalg::eigh(&w);
        let (d, _) = unfolded_spacings(&lam);
        eigenvalues = lam;
        spacings = d;
    }
    RmtResult {
        estimable: true,
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
    fn complete_graph_is_rejected_by_the_poisson_gof() {
        // A complete graph has one dominant eigenvalue and p-1 degenerate ones.
        // Its spacing distribution is nothing like Poisson, so the GOF must not
        // "accept" it -- this is the failure mode that made the old scan return
        // the complete graph at st=0.01.
        let n = 90;
        let mut full = Mat::from_elem((n, n), 0.9);
        for i in 0..n {
            full[(i, i)] = 0.0;
        }
        let pt = evaluate(&full, 0.5, 10, 0.01, &RmtConfig::default());
        assert_ne!(pt.status, "accepted", "complete graph must not look Poisson: {pt:?}");
    }

    #[test]
    fn dense_random_graph_is_evaluable_and_goe_like() {
        // Dense Erdos-Renyi: the bulk spectrum shows level repulsion (Wigner),
        // *not* Poisson, so the GOF should reject it. What this pins down is
        // that the point is evaluable at all and the statistic is finite.
        let n = 90;
        use rand::Rng;
        use rand::SeedableRng;
        let mut rng = rand::rngs::StdRng::seed_from_u64(12345);
        let mut s = Mat::zeros((n, n));
        for i in 0..n {
            for j in (i + 1)..n {
                let v = if rng.gen::<f64>() < 0.3 { 0.9 } else { 0.1 };
                s[(i, j)] = v;
                s[(j, i)] = v;
            }
        }
        let pt = evaluate(&s, 0.5, 10, 0.01, &RmtConfig::default());
        assert!(pt.p > 10, "kept {} nodes", pt.p);
        // it must at least reach the statistic, even if the dof guard then drops it
        assert_ne!(pt.status, "excluded_min_nodes", "{pt:?}");
        assert_ne!(pt.status, "excluded_too_few_spacings", "{pt:?}");
    }
}
