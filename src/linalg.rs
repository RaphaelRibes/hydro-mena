//! Dense linear algebra on row-major f64 matrices (rows = OTUs, cols = samples).
//!
//! These routines dominate runtime, so every O(n^2 m) pass is parallelised with
//! rayon over disjoint row blocks (`par_chunks_mut`, no shared writes), and the
//! symmetric eigenproblem goes through a Householder + implicit-QR solver
//! instead of the previous "find the largest off-diagonal element, rotate,
//! repeat" Jacobi loop: that rescanned the whole matrix on every rotation
//! (O(n^4) per decomposition) and ran at most 200 rotations **in total**, so any
//! matrix larger than roughly 20x20 came back essentially unconverged and said so
//! nowhere.

use ndarray::Array2;
use rayon::prelude::*;

pub type Mat = Array2<f64>;

/// Borrow the matrix as a contiguous row-major slice, materialising a standard
/// layout copy only when the input is a strided view (e.g. the transpose of the
/// trait table). The previous code called `as_slice().unwrap()` directly and
/// panicked on exactly that case.
fn as_std(x: &Mat) -> std::borrow::Cow<'_, Mat> {
    if x.as_slice().is_some() {
        std::borrow::Cow::Borrowed(x)
    } else {
        std::borrow::Cow::Owned(x.as_standard_layout().to_owned())
    }
}

/// Standardize rows of `x` to mean 0, sd 1 (Methods: x_ik = (y_ik - mean_i)/sd_i).
pub fn standardize_rows(x: &Mat) -> Mat {
    let (n, m) = (x.nrows(), x.ncols());
    if n == 0 || m == 0 {
        return x.clone();
    }
    let x = as_std(x);
    let x = x.as_ref();
    let mut z = x.clone();
    let xs = x.as_slice().expect("row-major matrix");
    let denom = m.saturating_sub(1).max(1) as f64;
    z.as_slice_mut()
        .expect("row-major matrix")
        .par_chunks_mut(m)
        .enumerate()
        .for_each(|(i, row)| {
            let src = &xs[i * m..(i + 1) * m];
            let mean = src.iter().sum::<f64>() / m as f64;
            let mut v = 0.0;
            for &y in src {
                let d = y - mean;
                v += d * d;
            }
            let sd = (v / denom).sqrt();
            if sd > 0.0 {
                for (dst, &y) in row.iter_mut().zip(src.iter()) {
                    *dst = (y - mean) / sd;
                }
            } else {
                row.fill(0.0);
            }
        });
    z
}

/// Per-OTU centred log-ratio transform (compositionality fix).
///
/// Abundance tables are compositional: the counts only carry information about
/// *ratios* between taxa, because they are constrained by the sequencing depth.
/// Correlating raw (or relative) abundances therefore picks up spurious
/// co-variation induced by that closure constraint. CLR removes it by mapping
/// every row (OTU) to its log-ratios against the row's geometric mean:
///
/// ```text
/// z_ij = log(x_ij + psi) - (1/m) * sum_k log(x_ik + psi)
/// ```
///
/// so `z_i` sums to zero by construction and only relative information survives.
/// `psi` is a pseudocount keeping zeros (and near-zeros) finite.
pub fn clr_transform(x: &Mat, pseudocount: f64) -> Mat {
    let (n, m) = (x.nrows(), x.ncols());
    if n == 0 || m == 0 {
        return x.clone();
    }
    let x = as_std(x);
    let x = x.as_ref();
    let mut z = x.clone();
    let xs = x.as_slice().expect("row-major matrix");
    let psi = pseudocount.max(0.0);
    z.as_slice_mut()
        .expect("row-major matrix")
        .par_chunks_mut(m)
        .enumerate()
        .for_each(|(i, row)| {
            let src = &xs[i * m..(i + 1) * m];
            // A negative pseudocount, or a value below -psi, would make the log
            // undefined and poison the whole matrix with NaN through the
            // downstream correlation, so clamp the shifted value to the smallest
            // positive double instead of propagating +/-inf.
            let log_of = |v: f64| (v + psi).max(f64::MIN_POSITIVE).ln();
            let mean = src.iter().map(|&v| log_of(v)).sum::<f64>() / m as f64;
            for (dst, &v) in row.iter_mut().zip(src.iter()) {
                *dst = log_of(v) - mean;
            }
        });
    z
}

/// Pearson correlation matrix of the rows of `x` (OTUs x samples), Eq. (1).
///
/// Computed from the standardized rows: with unit-variance rows,
/// r_ij = (1/(m-1)) * z_i . z_j, so the mean/sd pass happens once instead of
/// twice inside the O(n^2 m) double loop.
pub fn correlation_matrix(x: &Mat) -> Mat {
    let z = standardize_rows(x);
    let (n, m) = (z.nrows(), z.ncols());
    let mut r = Mat::zeros((n, n));
    if n == 0 || m < 2 {
        return r;
    }
    let zs = z.as_slice().expect("row-major matrix");
    let denom = (m - 1) as f64;
    r.as_slice_mut()
        .expect("row-major matrix")
        .par_chunks_mut(n)
        .enumerate()
        .for_each(|(i, row)| {
            let zi = &zs[i * m..(i + 1) * m];
            row[i] = 1.0;
            for j in (i + 1)..n {
                let zj = &zs[j * m..(j + 1) * m];
                let mut acc = 0.0;
                for k in 0..m {
                    acc += zi[k] * zj[k];
                }
                row[j] = acc / denom;
            }
        });
    mirror_upper(&mut r);
    r
}

/// Similarity matrix S = |R| with a zero diagonal (Eq. 2), straight from the
/// standardized rows, in one pass.
pub fn similarity_from_standardized(z: &Mat) -> Mat {
    let (n, m) = (z.nrows(), z.ncols());
    let mut s = Mat::zeros((n, n));
    if n == 0 || m < 2 {
        return s;
    }
    let zs = z.as_slice().expect("row-major matrix");
    let denom = (m - 1) as f64;
    s.as_slice_mut()
        .expect("row-major matrix")
        .par_chunks_mut(n)
        .enumerate()
        .for_each(|(i, row)| {
            let zi = &zs[i * m..(i + 1) * m];
            for j in (i + 1)..n {
                let zj = &zs[j * m..(j + 1) * m];
                let mut acc = 0.0;
                for k in 0..m {
                    acc += zi[k] * zj[k];
                }
                row[j] = (acc / denom).abs();
            }
        });
    mirror_upper(&mut s);
    s
}

/// **Signed** similarity matrix S = r with a zero diagonal.
///
/// [`similarity_from_standardized`] throws the sign away (|r|, the paper's
/// Eq. 2 and the default). That is fine for the RMT threshold and the network
/// topology, but it collapses competition (negative association) and
/// cooperation (positive) onto the same number. This variant keeps the raw
/// Pearson r; callers that need the paper-faithful non-negative matrix take
/// the absolute value afterwards.
pub fn similarity_signed(z: &Mat) -> Mat {
    let (n, m) = (z.nrows(), z.ncols());
    let mut s = Mat::zeros((n, n));
    if n == 0 || m < 2 {
        return s;
    }
    let zs = z.as_slice().expect("row-major matrix");
    let denom = (m - 1) as f64;
    s.as_slice_mut()
        .expect("row-major matrix")
        .par_chunks_mut(n)
        .enumerate()
        .for_each(|(i, row)| {
            let zi = &zs[i * m..(i + 1) * m];
            for j in (i + 1)..n {
                let zj = &zs[j * m..(j + 1) * m];
                let mut acc = 0.0;
                for k in 0..m {
                    acc += zi[k] * zj[k];
                }
                row[j] = acc / denom;
            }
        });
    mirror_upper(&mut s);
    s
}

/// Copy the upper triangle into the lower one (single flat pass, no aliasing).
fn mirror_upper(a: &mut Mat) {
    let n = a.nrows();
    let d = a.as_slice_mut().expect("row-major matrix");
    for i in 1..n {
        let (head, tail) = d.split_at_mut(i * n);
        let row = &mut tail[..n];
        for j in 0..i {
            row[j] = head[j * n + i];
        }
    }
}

/// Symmetric eigendecomposition. Returns eigenvalues **ascending** with the
/// matching eigenvectors as columns.
pub fn eigh(a: &Mat) -> (Vec<f64>, Mat) {
    let n = a.nrows();
    if n == 0 {
        return (Vec::new(), Mat::zeros((0, 0)));
    }
    let m = nalgebra::DMatrix::from_row_slice(n, n, a.as_slice().expect("row-major matrix"));
    let eig = nalgebra::SymmetricEigen::new(m);
    // nalgebra does not promise an ordering, so sort explicitly: ascending
    // eigenvalues with the eigenvectors permuted to match.
    let mut idx: Vec<usize> = (0..n).collect();
    let raw: Vec<f64> = eig.eigenvalues.iter().copied().collect();
    idx.sort_by(|&i, &j| raw[i].partial_cmp(&raw[j]).unwrap());
    let vals: Vec<f64> = idx.iter().map(|&i| raw[i]).collect();
    let mut vecs = Mat::zeros((n, n));
    let vs = eig.eigenvectors.as_slice();
    for (new_j, &old_j) in idx.iter().enumerate() {
        for i in 0..n {
            vecs[(i, new_j)] = vs[old_j * n + i];
        }
    }
    (vals, vecs)
}

/// SVD via the eigendecomposition of X^T X (m x m, m = samples, small).
/// Returns (U, s, Vt) with singular values descending — used for module
/// eigengenes, Eq. (10).
pub fn svd(x: &Mat) -> (Mat, Vec<f64>, Mat) {
    let (n, m) = (x.nrows(), x.ncols());
    if n == 0 || m == 0 {
        return (Mat::zeros((n, 0)), Vec::new(), Mat::zeros((0, m)));
    }
    let xt = x.t().to_owned();
    let xtx = xt.dot(x);
    let (evals, v) = eigh(&xtx);
    let k = m.min(n);
    let mut s: Vec<f64> = evals.iter().rev().take(k).map(|e| e.max(0.0).sqrt()).collect();
    let mut vt = Mat::zeros((k, m));
    for j in 0..k {
        for c in 0..m {
            vt[(j, c)] = v[(c, m - 1 - j)];
        }
    }
    // U = X V S^-1
    let mut u = Mat::zeros((n, k));
    for j in 0..k {
        if s[j] > 1e-12 {
            for i in 0..n {
                let mut acc = 0.0;
                for c in 0..m {
                    acc += x[(i, c)] * vt[(j, c)];
                }
                u[(i, j)] = acc / s[j];
            }
        }
    }
    s.truncate(k);
    (u, s, vt)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn eigh_known_spectrum() {
        // eigenvalues of [[2,1],[1,2]] are 1 and 3
        let a = Mat::from_shape_vec((2, 2), vec![2.0, 1.0, 1.0, 2.0]).unwrap();
        let (vals, vecs) = eigh(&a);
        assert!((vals[0] - 1.0).abs() < 1e-10, "got {vals:?}");
        assert!((vals[1] - 3.0).abs() < 1e-10, "got {vals:?}");
        for i in 0..2 {
            let lhs: f64 = (0..2).map(|j| a[(i, j)] * vecs[(j, 0)]).sum();
            assert!((lhs - vals[0] * vecs[(i, 0)]).abs() < 1e-10);
        }
    }

    #[test]
    fn eigh_diagonal_is_ascending() {
        let a = Mat::from_diag(&ndarray::arr1(&[3.0, -1.0, 2.0]));
        let (vals, _) = eigh(&a);
        assert_eq!(vals.len(), 3);
        assert!((vals[0] + 1.0).abs() < 1e-12);
        assert!((vals[1] - 2.0).abs() < 1e-12);
        assert!((vals[2] - 3.0).abs() < 1e-12);
    }

    #[test]
    fn clr_is_centred_and_matches_known_values() {
        // psi = 0 on rows [2,4,8] and [5,5,5]: the log-ratios are exact
        // multiples of ln 2, and the constant row must collapse to exactly zero.
        let x = Mat::from_shape_vec((2, 3), vec![2.0, 4.0, 8.0, 5.0, 5.0, 5.0]).unwrap();
        let z = clr_transform(&x, 0.0);
        let ln2 = 2.0f64.ln();
        for (j, want) in [-ln2, 0.0, ln2].iter().enumerate() {
            assert!((z[(0, j)] - want).abs() < 1e-12, "row0 = {:?}", z.row(0));
        }
        for j in 0..3 {
            assert_eq!(z[(1, j)], 0.0, "constant row must be exactly zero");
        }
        // pseudocount shifts the geometric mean: psi = 1 on [1,3] gives
        // log 2 - (log 2 + log 4)/2 = -0.5 ln 2, and +0.5 ln 2 for the other cell.
        let y = Mat::from_shape_vec((1, 2), vec![1.0, 3.0]).unwrap();
        let zy = clr_transform(&y, 1.0);
        assert!((zy[(0, 0)] + 0.5 * ln2).abs() < 1e-12, "got {:?}", zy.row(0));
        assert!((zy[(0, 1)] - 0.5 * ln2).abs() < 1e-12, "got {:?}", zy.row(0));
        // every row sums to zero by construction
        for i in 0..2 {
            assert!(z.row(i).sum().abs() < 1e-12);
        }
    }

    #[test]
    fn correlation_of_identical_rows_is_one() {
        let x = Mat::from_shape_vec((2, 4), vec![1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0]).unwrap();
        let r = correlation_matrix(&x);
        assert!((r[(0, 1)] - 1.0).abs() < 1e-12);
        assert!((r[(1, 0)] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn similarity_matches_correlation_and_has_zero_diagonal() {
        let x = Mat::from_shape_vec(
            (3, 4),
            vec![1.0, 2.0, 3.0, 4.0, 4.0, 3.0, 2.0, 1.0, 0.0, 1.0, 5.0, 2.0],
        )
        .unwrap();
        let z = standardize_rows(&x);
        let s = similarity_from_standardized(&z);
        let r = correlation_matrix(&x);
        for i in 0..3 {
            assert_eq!(s[(i, i)], 0.0);
            for j in 0..3 {
                if i != j {
                    assert!((s[(i, j)] - r[(i, j)].abs()).abs() < 1e-12);
                }
            }
        }
    }

    #[test]
    fn signed_similarity_keeps_anticorrelation() {
        // rows [1,2,3,4] and [4,3,2,1]: Pearson r = -1
        let x = Mat::from_shape_vec((2, 4), vec![1.0, 2.0, 3.0, 4.0, 4.0, 3.0, 2.0, 1.0]).unwrap();
        let z = standardize_rows(&x);
        let r = similarity_signed(&z);
        assert_eq!(r[(0, 0)], 0.0);
        assert_eq!(r[(1, 1)], 0.0);
        assert!((r[(0, 1)] + 1.0).abs() < 1e-12, "got {}", r[(0, 1)]);
        assert!((r[(1, 0)] + 1.0).abs() < 1e-12, "asymmetric: {}", r[(1, 0)]);
        // signed value is exactly the correlation, and the default keeps |r|
        let c = correlation_matrix(&x);
        assert!((r[(0, 1)] - c[(0, 1)]).abs() < 1e-12);
        let s = similarity_from_standardized(&z);
        assert!((s[(0, 1)] - 1.0).abs() < 1e-12);
        // cooperative rows stay positive: sign is not just flipped
        let y = Mat::from_shape_vec((2, 4), vec![1.0, 2.0, 3.0, 4.0, 2.0, 4.0, 6.0, 8.0]).unwrap();
        let ry = similarity_signed(&standardize_rows(&y));
        assert!((ry[(0, 1)] - 1.0).abs() < 1e-12, "got {}", ry[(0, 1)]);
    }
}
