//! RMT-based automatic threshold detection (Deng et al. 2012, Fig. 2, steps a-k).
//!
//! a-b  similarity S = |Pearson r|
//! c-d  adjacency A_ij = 1 when S_ij >= st, zero diagonal, drop degree-0 nodes
//! e    eigen-decompose the *weighted* thresholded similarity matrix of the
//!      giant component (paper Eq. 3: keep the weights s_ij, zero the
//!      sub-threshold entries, zero diagonal — binarising gives repeated
//!      eigenvalues and nothing for the spacing test to resolve)
//! f    unfold by **local density average**: estimate the local mean spacing
//!      from a symmetric window of neighbours and integrate dλ/ρ(λ). This is
//!      positive by construction, so there is no polynomial fit and no clipping
//! g-i  spacings rescaled to unit mean
//! j-k  binned chi^2 goodness-of-fit of the spacing histogram against the
//!      Poisson law exp(-s) at alpha
//! l    continuous diagnostics alongside the binary verdict: the MLE Brody
//!      exponent (0 = Poisson, 1 = GOE) and the KS distance to the Wigner
//!      surmise, so a near-miss is readable as a number instead of "rejected".
//!      Above 512 giant-component nodes the scan takes the spectrum from
//!      Lanczos (full reorthogonalisation) instead of the dense O(n^3) solver.
//!
//! Every candidate threshold is reported with a status (`accepted`,
//! `rejected_gof`, `excluded_*`). A point is only eligible when the graph keeps
//! at least `min_kept_frac` of the OTUs and at least `min_giant_frac` of the
//! kept nodes fall in the giant component; both floors are 0.5, the point below
//! which the calibration pilot showed modularity recovery collapsing. When no
//! candidate passes the GOF the result is *not estimable*: st is NaN and no
//! adjacency is built — only the audit trail (one row per candidate) is
//! written, so a failure can be diagnosed instead of being papered over with a
//! fallback graph.
use crate::linalg::{self, Mat};
use crate::stats;
use rayon::prelude::*;

/// Giant components larger than this are eigen-decomposed with Lanczos instead
/// of the dense solver: above a few hundred nodes the O(n^3) decomposition is
/// the entire runtime of the pipeline, while the spacing GOF only needs a
/// representative stretch of eigenvalues.
const LANCZOS_MIN_N: usize = 512;

/// Number of Lanczos steps for an n x n matrix: enough to converge the extremal
/// Ritz values well past the 5th, capped so the cost stays O(60 n^2).
fn lanczos_steps(n: usize) -> usize {
    n.saturating_sub(2).clamp(2, 60)
}

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
    /// minimum share of OTUs that must survive the threshold. 0.5 is the point
    /// below which the calibration pilot showed ARI collapsing (< 0.35), so a
    /// sparser graph is not a usable estimate of community structure.
    pub min_kept_frac: f64,
    /// minimum share of kept nodes inside the giant component
    pub min_giant_frac: f64,
    /// seed for the Lanczos start vector (only used above `LANCZOS_MIN_N`);
    /// fixed so a scan is reproducible
    pub seed: u64,
}

impl Default for RmtConfig {
    fn default() -> Self {
        Self {
            giant: true,
            trim_extremes: true,
            min_spacings: 20,
            min_dof: 2,
            min_kept_frac: 0.5,
            min_giant_frac: 0.5,
            seed: 42,
        }
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
    /// MLE Brody exponent of the spacings: 0 = Poisson, 1 = GOE. Continuous
    /// version of the accept/reject verdict (NaN when there are no spacings).
    pub brody_beta: f64,
    /// Kolmogorov-Smirnov distance to the Wigner surmise (GOE). Read against
    /// `ks`: the smaller one names the ensemble the spacing distribution
    /// actually resembles (NaN when there are no spacings).
    pub ks_goe: f64,
    /// share of kept nodes inside the giant component the test was run on
    pub giant_frac: f64,
    /// why this candidate was accepted or dropped: `accepted`, `rejected_gof`,
    /// `excluded_min_nodes`, `excluded_tiny_giant`, `excluded_too_few_spacings`,
    /// `excluded_low_dof`. Every candidate is reported, so the curve can be
    /// audited instead of silently thinned to its survivors.
    pub status: String,
}

/// Symmetric 0/1 adjacency from the similarity matrix at threshold `st`.
///
/// Requires `s >= 0` in the entries that matter: the similarity handed in is
/// always |r|. `--signed` keeps the sign of r elsewhere (edge metadata and
/// `similarity.csv`), not here: Deng Eq. 3 and the Poisson spacing GOF assume
/// non-negative weights, so a negative entry would be silently thresholded as
/// if it were its absolute value anyway — better to make the abs explicit in
/// the caller than to leave a threshold rule that reads as a bug.
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
/// Only used by `polyfit_recovers_a_line`; the scan itself now unfolds by local
/// density average, so this is test-only code.
#[cfg(test)]
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

#[cfg(test)]
fn polyval(coef: &[f64], x: f64) -> f64 {
    let mut acc = 0.0;
    for &c in coef.iter().rev() {
        acc = acc * x + c;
    }
    acc
}

/// Unfold the spectrum by **local density average** (paper step (f): "fitting
/// the original integrated density to a cubic spline or by local density
/// average").
///
/// For each eigenvalue, estimate the local mean spacing from a symmetric window
/// of w neighbours and integrate dλ/ρ(λ). This is positive by construction, so
/// there is no monotone clipping and therefore no plateau of zero spacings —
/// the previous degree-5 polynomial plus clipping produced exactly that, and it
/// was the real cause of every `excluded_too_few_spacings` in the audit trail.
fn unfolded_spacings(lam_in: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let mut sorted: Vec<f64> = lam_in.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let lam = &sorted;
    let p = lam.len();
    if p < 6 {
        return (Vec::new(), lam.clone());
    }
    let w = (p / 40).max(2).min((p / 4).max(2));
    let mut xi = vec![0.0f64; p];
    for i in 1..p {
        let lo = i.saturating_sub(w);
        let hi = (i + w).min(p - 1);
        let span = (lam[hi] - lam[lo]).abs();
        let local = span / (hi - lo).max(1) as f64;
        if local > 0.0 && local.is_finite() {
            xi[i] = xi[i - 1] + (lam[i] - lam[i - 1]) / local;
        } else {
            // repeated eigenvalue: no resolvable level spacing here
            xi[i] = xi[i - 1];
        }
    }
    let mut d: Vec<f64> = (0..p - 1).map(|i| xi[i + 1] - xi[i]).collect();
    let mean = d.iter().sum::<f64>() / d.len() as f64;
    if !(mean > 0.0) {
        return (Vec::new(), xi);
    }
    for v in d.iter_mut() {
        *v /= mean;
    }
    (d, xi)
}

/// Eigenvalues of the largest (and smallest) part of the spectrum by Lanczos
/// iteration with **full reorthogonalisation**, in ascending order.
///
/// Used by the scan above `LANCZOS_MIN_N` nodes, where the dense O(n^3)
/// decomposition dominates the pipeline. `k` steps cost O(k n^2) mat-vecs and
/// converge the extremal Ritz values; the returned set is the *upper* part of
/// the spectrum, not all n eigenvalues, so a spacing test run on it reads the
/// well-resolved tail rather than the bulk. That is an approximation and is
/// documented as one: below the cutoff the dense solver is used, and the scan
/// point is reported with the same guards either way.
///
/// Full reorthogonalisation (twice, against every previous basis vector) is
/// what keeps the Ritz values accurate to ~1e-10 instead of degrading to the
/// classic three-term recurrence's ghost-eigenvalue behaviour.
fn lanczos_eigvals(w: &Mat, k: usize, seed: u64) -> Vec<f64> {
    use rand::{Rng, SeedableRng};
    let n = w.nrows();
    if n < 2 || w.ncols() != n {
        return Vec::new();
    }
    let m = k.clamp(1, n);
    let ws = w.as_slice().expect("row-major matrix");
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let mut q_cur: Vec<f64> = (0..n).map(|_| rng.gen::<f64>() - 0.5).collect();
    let mut nq = q_cur.iter().map(|v| v * v).sum::<f64>().sqrt();
    if !(nq > 0.0) || !nq.is_finite() {
        // degenerate draw: fall back to a coordinate vector
        q_cur.iter_mut().for_each(|v| *v = 0.0);
        q_cur[0] = 1.0;
        nq = 1.0;
    }
    for v in q_cur.iter_mut() {
        *v /= nq;
    }

    let dot = |a: &[f64], b: &[f64]| a.iter().zip(b).map(|(x, y)| x * y).sum::<f64>();
    let norm = |a: &[f64]| dot(a, a).sqrt();

    let mut basis: Vec<Vec<f64>> = vec![q_cur.clone()];
    // The Rayleigh quotient Q^T A Q, accumulated from the reorthogonalisation
    // coefficients: with full reorthogonalisation this is the exact projection
    // onto the computed subspace, so its eigenvalues (Ritz values) are optimal
    // there. Keeping only the three-term (tridiagonal) entries would discard
    // the corrections that make the basis orthonormal in finite precision.
    let mut t = Mat::zeros((m, m));
    let mut used = 0usize;
    let mut scale = 1.0f64;

    for step in 0..m {
        // z = A q_j. The three-term recurrence's beta_{j-1} q_{j-1} term is
        // not subtracted explicitly: q_{j-1} is in `basis`, so the
        // reorthogonalisation below removes that component anyway.
        let mut z = vec![0.0f64; n];
        for i in 0..n {
            let row = &ws[i * n..(i + 1) * n];
            let mut acc = 0.0;
            for j in 0..n {
                acc += row[j] * q_cur[j];
            }
            z[i] = acc;
        }
        let a = dot(&q_cur, &z);
        for i in 0..n {
            z[i] -= a * q_cur[i];
        }
        t[(step, step)] = a;
        // full reorthogonalisation, two passes (the second catches the loss of
        // orthogonality the first pass itself introduces); each coefficient is
        // the corresponding entry of Q^T A Q
        for _pass in 0..2 {
            for (i, v) in basis.iter().enumerate() {
                let c = dot(v, &z);
                if c != 0.0 {
                    t[(i, step)] += c;
                    t[(step, i)] += c;
                    for k in 0..n {
                        z[k] -= c * v[k];
                    }
                }
            }
        }
        used = step + 1;
        if step == 0 {
            scale = a.abs().max(1.0);
        }
        let b = norm(&z);
        if step + 1 == m || b <= 1e-12 * scale {
            break;
        }
        for i in 0..n {
            z[i] /= b;
        }
        q_cur = z;
        basis.push(q_cur.clone());
    }

    if used == 0 {
        return Vec::new();
    }
    let t = t.slice(ndarray::s![..used, ..used]).to_owned();
    linalg::eigh(&t).0
}

/// Largest connected component of the binary graph over `keep`, as positions
/// into `keep`, plus its fraction of all kept nodes. A disconnected graph is
/// Poisson-like for free, so the spectral test is run on the giant component.
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
        brody_beta: f64::NAN,
        ks_goe: f64::NAN,
        giant_frac,
        status: reason.to_string(),
    };
    if p < min_nodes.max(6) || (p as f64) < cfg.min_kept_frac * s.nrows() as f64 {
        return ex(p, edges, "excluded_min_nodes", f64::NAN);
    }
    let (giant, giant_frac) = if cfg.giant {
        giant_component(&a, &keep)
    } else {
        ((0..keep.len()).collect::<Vec<usize>>(), 1.0)
    };
    if giant.len() < min_nodes.max(6) || giant_frac < cfg.min_giant_frac {
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
    let mut lam = if g > LANCZOS_MIN_N {
        // The dense O(g^3) decomposition is the whole runtime of the pipeline
        // at this size; the spacing GOF needs a stretch of eigenvalues, not all
        // of them. Seed per threshold so a rescan is reproducible.
        let seed = cfg.seed ^ (st * 1e6).round() as u64;
        lanczos_eigvals(&w, lanczos_steps(g), seed)
    } else {
        linalg::eigh(&w).0
    };
    if lam.len() < 2 {
        return ex(p, edges, "excluded_too_few_spacings", giant_frac);
    }
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
    // continuous diagnostics, reported for every point that has spacings --
    // they are what makes a near-miss legible instead of just "rejected"
    let brody_beta = stats::brody_beta(&d);
    let ks_goe = stats::ks_goe(&d);
    let mut pt = ScanPoint {
        st,
        p,
        edges,
        chi2,
        dof,
        crit,
        poisson: false,
        ks,
        brody_beta,
        ks_goe,
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
        // Dense even above LANCZOS_MIN_N: this path produces the spectrum that
        // spectrum.csv and the RMT figure are drawn from, so it stays exact.
        // The Lanczos approximation is confined to the scan decision.
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
        // and the continuous diagnostics must be filled, not left at zero/NaN
        assert!(pt.brody_beta.is_finite() && pt.ks_goe.is_finite(), "{pt:?}");
        assert!((0.0..=2.0).contains(&pt.brody_beta), "{pt:?}");
        // excluded points have no spectrum, so they must say NaN, not 0
        let none = evaluate(&s, 0.999, 10, 0.01, &RmtConfig::default());
        assert_eq!(none.status, "excluded_min_nodes", "{none:?}");
        assert!(none.brody_beta.is_nan() && none.ks_goe.is_nan(), "{none:?}");
    }

    #[test]
    fn lanczos_matches_eigh_on_the_extremal_spectrum() {
        // Random symmetric 600x600 with the spectral structure the scan
        // actually sees: a few separated signal eigenvalues over a noise bulk
        // (a spiked model -- modules plus noise). Extremal Lanczos convergence
        // is governed by the relative gap, so 60 steps reproduce these to
        // machine precision.
        //
        // A *dense* U(-1,1) matrix would be a worse test, not a harder one:
        // its top five eigenvalues are pairwise as close as 0.07 on a spectral
        // width of 56, so no 60-step Lanczos reaches 1e-6 there (measured:
        // 1e-4 for the 3rd/4th). That regime is covered by the top-1 check
        // below, which is the accuracy that case can honestly support.
        let n = 600;
        use rand::Rng;
        use rand::SeedableRng;
        let mut rng = rand::rngs::StdRng::seed_from_u64(2024);
        let mut u = vec![vec![0.0f64; n]; 5];
        for r in 0..5 {
            let mut nrm = 0.0;
            for i in 0..n {
                let v: f64 = rng.gen::<f64>() * 2.0 - 1.0;
                u[r][i] = v;
                nrm += v * v;
            }
            let nrm = nrm.sqrt();
            for i in 0..n {
                u[r][i] /= nrm;
            }
        }
        let amp = [10.0, 7.0, 5.0, 3.5, 2.5];
        let mut a = Mat::zeros((n, n));
        for i in 0..n {
            for j in i..n {
                let mut acc = 0.0;
                for r in 0..5 {
                    acc += amp[r] * u[r][i] * u[r][j];
                }
                if i != j {
                    acc += 0.01 * (rng.gen::<f64>() * 2.0 - 1.0);
                }
                a[(i, j)] = acc;
                a[(j, i)] = acc;
            }
        }

        let (full, _) = linalg::eigh(&a);
        let mut ritz = lanczos_eigvals(&a, lanczos_steps(n), 7);
        assert_eq!(lanczos_steps(n), 60);
        assert_eq!(ritz.len(), 60, "expected 60 Ritz values");
        ritz.sort_by(|x, y| x.partial_cmp(y).unwrap());
        for t in 0..5 {
            let want = full[n - 1 - t];
            let got = ritz[ritz.len() - 1 - t];
            let rel = (got - want).abs() / want.abs().max(1.0);
            assert!(rel < 1e-6, "Ritz[{t}] = {got}, eigh = {want}, rel = {rel:e}");
        }

        // dense U(-1,1): only the leading (well-separated) Ritz value can be
        // demanded at this step count
        let mut b = Mat::zeros((n, n));
        for i in 0..n {
            for j in (i + 1)..n {
                let v = rng.gen::<f64>() * 2.0 - 1.0;
                b[(i, j)] = v;
                b[(j, i)] = v;
            }
        }
        let (full, _) = linalg::eigh(&b);
        let mut ritz = lanczos_eigvals(&b, lanczos_steps(n), 7);
        ritz.sort_by(|x, y| x.partial_cmp(y).unwrap());
        let want = full[n - 1];
        let got = ritz[ritz.len() - 1];
        assert!((got - want).abs() / want.abs() < 1e-6, "top Ritz {got} vs eigh {want}");
    }

    #[test]
    fn large_graph_takes_the_lanczos_path_and_stays_estimable() {
        // A 560-node block graph is above the cutoff, so evaluate() must route
        // through Lanczos and still produce a finite point with diagnostics.
        let n = 560;
        use rand::Rng;
        use rand::SeedableRng;
        let mut rng = rand::rngs::StdRng::seed_from_u64(99);
        let mut s = Mat::zeros((n, n));
        let block = 40;
        for i in 0..n {
            for j in (i + 1)..n {
                let same = i / block == j / block;
                let p = if same { 0.6 } else { 0.005 };
                let v = if rng.gen::<f64>() < p { 0.8 } else { 0.1 };
                s[(i, j)] = v;
                s[(j, i)] = v;
            }
        }
        let pt = evaluate(&s, 0.5, 10, 0.01, &RmtConfig::default());
        assert!(pt.p > 500, "kept {} nodes", pt.p);
        assert_ne!(pt.status, "excluded_too_few_spacings", "{pt:?}");
        assert!(pt.brody_beta.is_finite() && pt.ks_goe.is_finite(), "{pt:?}");
    }
}
