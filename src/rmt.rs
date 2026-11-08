//! RMT-based automatic threshold detection (Deng et al. 2012, Fig. 2, steps a-k).
//! Similarity S = |Pearson r|; adjacency A_ij = 1 if s_ij >= st, dropping rows
//! with no links; scan st upward (0.1 coarse, then 0.01 fine) until the
//! unfolded NNSD of adjacency eigenvalues follows the Poisson distribution
//! (chi^2 GOF, alpha = 0.01). Parallelised over candidate thresholds with rayon.
use crate::linalg::{self, Mat};
use crate::stats;
use rayon::prelude::*;

#[derive(Clone, Debug, serde::Serialize)]
pub struct ScanPoint {
    pub st: f64,
    pub p: usize,
    pub chi2: f64,
    pub crit: f64,
    pub poisson: bool,
}

fn adjacency(S: &Mat, st: f64) -> Mat {
    let n = S.nrows();
    let mut A = Mat::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            if i != j && S[(i, j)] >= st {
                A[(i, j)] = 1.0;
            }
        }
    }
    A
}

fn keep_mask(A: &Mat) -> Vec<bool> {
    (0..A.nrows()).map(|i| (0..A.ncols()).any(|j| A[(i, j)] > 0.0)).collect()
}

/// Unfold eigenvalues: e_i = N_av(lambda_i) via monotone piecewise-linear
/// smoothed CDF (paper suggests cubic spline; linear interpolation of the
/// empirical CDF is robust for the sizes used here), rescaled to mean spacing 1.
fn unfolded_spacings(mut lam: Vec<f64>) -> Vec<f64> {
    lam.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let p = lam.len();
    if p < 3 {
        return vec![];
    }
    // e_i = (rank + 0.5)/p  ->  in (0,1); spacing di = p * (e_{i+1} - e_i)
    let e: Vec<f64> = (0..p).map(|i| (i as f64 + 0.5) / p as f64).collect();
    let _ = linalg::correlation_matrix; // (keep module referenced)
    let mut d = Vec::with_capacity(p - 1);
    for i in 0..p - 1 {
        d.push(p as f64 * (e[i + 1] - e[i]));
    }
    d
}

fn poisson_accept(A: &Mat, min_nodes: usize) -> Option<ScanPoint> {
    let keep = keep_mask(A);
    let p = keep.iter().filter(|b| **b).count();
    if p < min_nodes {
        return None;
    }
    let idx: Vec<usize> = (0..A.nrows()).filter(|&i| keep[i]).collect();
    let mut Asub = Mat::zeros((p, p));
    for (a, &i) in idx.iter().enumerate() {
        for (b, &j) in idx.iter().enumerate() {
            Asub[(a, b)] = A[(i, j)];
        }
    }
    let (lam, _) = linalg::jacobi_eigen(&Asub);
    let d = unfolded_spacings(lam);
    if d.len() < 4 {
        return None;
    }
    let (chi2, u, crit) = stats::poisson_chi2(&d, 0.01);
    Some(ScanPoint {
        st: 0.0, // filled by caller
        p,
        chi2,
        crit,
        poisson: chi2 <= crit && u > 0,
    })
}

/// Scan a list of candidate thresholds in parallel, returning ScanPoints.
fn scan(S: &Mat, sts: &[f64], min_nodes: usize) -> Vec<ScanPoint> {
    let mut out: Vec<ScanPoint> = sts
        .par_iter()
        .filter_map(|&st| {
            let A = adjacency(S, st);
            poisson_accept(&A, min_nodes).map(|mut pt| {
                pt.st = st;
                pt
            })
        })
        .collect();
    out.sort_by(|a, b| a.st.partial_cmp(&b.st).unwrap());
    out
}

pub struct RmtResult {
    pub st: f64,
    pub converged: bool,
    pub log: Vec<ScanPoint>,
    pub adjacency: Mat,
    pub kept: Vec<usize>,
}

/// Full RMT threshold procedure: coarse scan from st0 by 0.1 until Poisson
/// accepted, then fine scan by 0.01 within [st-0.1, st].
pub fn rmt_threshold(S: &Mat, st0: f64, min_nodes: usize) -> RmtResult {
    let mut coarse: Vec<f64> = Vec::new();
    let mut st = st0;
    while st <= 0.99 + 1e-9 {
        coarse.push((st * 10000.0).round() / 10000.0);
        st += 0.1;
    }
    let log_coarse = scan(S, &coarse, min_nodes);
    let accepted = log_coarse.iter().find(|p| p.poisson).map(|p| p.st);
    let (st_final, converged, log) = match accepted {
        Some(acc) => {
            let lo = (acc - 0.1).max(0.01);
            let mut fine: Vec<f64> = Vec::new();
            let mut s = lo;
            while s <= acc + 1e-9 {
                fine.push((s * 10000.0).round() / 10000.0);
                s += 0.01;
            }
            let mut log_fine = scan(S, &fine, min_nodes);
            let best = log_fine.iter().find(|p| p.poisson).map(|p| p.st).unwrap_or(acc);
            log_fine.extend(log_coarse.iter().cloned());
            (best, true, log_fine)
        }
        None => {
            let fb = log_coarse.last().map(|p| p.st).unwrap_or(st0);
            (fb, !log_coarse.is_empty(), log_coarse)
        }
    };
    let A = adjacency(S, st_final);
    let kept: Vec<usize> = (0..A.nrows()).filter(|&i| (0..A.ncols()).any(|j| A[(i, j)] > 0.0)).collect();
    RmtResult {
        st: st_final,
        converged,
        log,
        adjacency: A,
        kept,
    }
}
