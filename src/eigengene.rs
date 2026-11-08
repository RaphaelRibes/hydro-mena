//! Module eigengenes (Eq. 10-12): SVD of each module's standardized abundance
//! submatrix; E_b = first right singular vector; variance explained Phi_b;
//! module membership MM_iE = cor(x_i, E_b); eigengene-vs-trait correlations
//! with r and p (this is the data behind paper Figure 5).
use crate::linalg::{self, Mat};
use crate::stats;

pub struct EigengeneResult {
    /// rows = modules (0..K-1), cols = samples
    pub E: Mat,
    pub var_explained: Vec<f64>,
    /// rows = modules, cols = traits: r and p
    pub trait_r: Mat,
    pub trait_p: Mat,
    pub mmie: Vec<f64>, // per node (input order)
}

pub fn eigengenes(z: &Mat, module_of: &[usize], n_modules: usize, trait_z: &Mat) -> EigengeneResult {
    let n = z.nrows();
    let m = z.ncols();
    let t = trait_z.ncols();
    let mut E = Mat::zeros((n_modules, m));
    let mut var_explained = vec![0.0; n_modules];
    let mut eigvec: Vec<Vec<f64>> = vec![Vec::new(); n_modules];
    for b in 0..n_modules {
        let members: Vec<usize> = (0..n).filter(|&i| module_of[i] == b).collect();
        if members.is_empty() {
            continue;
        }
        let xb = Mat::from_shape_fn((members.len(), m), |(i, j)| z[(members[i], j)]);
        let (_, s, vt) = linalg::svd(&xb);
        let row = vt.row(0).to_owned();
        for j in 0..m {
            E[(b, j)] = row[j];
        }
        eigvec[b] = row.to_vec();
        let ss: f64 = s.iter().map(|x| x * x).sum();
        var_explained[b] = if ss > 0.0 { s[0] * s[0] / ss } else { 1.0 };
    }
    // eigengene vs standardized traits -> r, p
    let mut trait_r = Mat::zeros((n_modules, t));
    let mut trait_p = Mat::zeros((n_modules, t));
    for b in 0..n_modules {
        let e: Vec<f64> = (0..m).map(|j| E[(b, j)]).collect();
        for h in 0..t {
            let th: Vec<f64> = (0..m).map(|j| trait_z[(j, h)]).collect();
            let (r, p) = stats::pearson(&e, &th);
            trait_r[(b, h)] = r;
            trait_p[(b, h)] = p;
        }
    }
    // module membership MM_iE = cor(x_i, E_{b(i)})
    let mut mmie = vec![0.0; n];
    for i in 0..n {
        let b = module_of[i];
        let xi: Vec<f64> = (0..m).map(|j| z[(i, j)]).collect();
        let (r, _) = stats::pearson(&xi, &eigvec[b]);
        mmie[i] = r;
    }
    EigengeneResult {
        E,
        var_explained,
        trait_r,
        trait_p,
        mmie,
    }
}

/// Trait-based OTU significance GS_i,h = cor(x_i, T_h)^2 (Eq. 13), traits
/// standardized across samples first.
pub fn trait_significance(z: &Mat, traits: &Mat) -> Mat {
    let n = z.nrows();
    let m = z.ncols();
    let t = traits.ncols();
    let tz = linalg::standardize_rows(&traits.t().to_owned());
    let mut gs = Mat::zeros((n, t));
    for i in 0..n {
        let xi: Vec<f64> = (0..m).map(|j| z[(i, j)]).collect();
        for h in 0..t {
            let th: Vec<f64> = (0..m).map(|j| tz[(h, j)]).collect();
            let (r, _) = stats::pearson(&xi, &th);
            gs[(i, h)] = r * r;
        }
    }
    gs
}
