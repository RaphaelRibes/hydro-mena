//! Dense linear algebra on row-major f64 matrices (Vec<f64>, shape (rows, cols)).
use ndarray::Array2;

pub type Mat = Array2<f64>;

/// Pearson correlation matrix of rows of `x` (OTUs x samples), Eq. (1).
pub fn correlation_matrix(x: &Mat) -> Mat {
    let n = x.nrows();
    let m = x.ncols();
    let mut mean = vec![0.0; n];
    let mut sd = vec![0.0; n];
    for i in 0..n {
        let mut s = 0.0;
        for j in 0..m {
            s += x[(i, j)];
        }
        mean[i] = s / m as f64;
        let mut v = 0.0;
        for j in 0..m {
            let d = x[(i, j)] - mean[i];
            v += d * d;
        }
        sd[i] = (v / (m - 1) as f64).sqrt();
    }
    let mut R = Mat::zeros((n, n));
    for i in 0..n {
        for j in i..n {
            let mut cov = 0.0;
            for k in 0..m {
                cov += (x[(i, k)] - mean[i]) * (x[(j, k)] - mean[j]);
            }
            let r = if sd[i] > 0.0 && sd[j] > 0.0 {
                cov / ((m - 1) as f64 * sd[i] * sd[j])
            } else {
                0.0
            };
            R[(i, j)] = r;
            R[(j, i)] = r;
        }
    }
    R
}

/// Symmetric eigen-decomposition via the Jacobi method. Returns eigenvalues
/// ascending with matching eigenvectors as columns.
pub fn jacobi_eigen(a: &Mat) -> (Vec<f64>, Mat) {
    let n = a.nrows();
    let mut a = a.clone();
    let mut v = Mat::eye(n);
    for _ in 0..200 {
        // find largest off-diagonal
        let mut p = 0;
        let mut q = 1;
        let mut max = 0.0f64;
        for i in 0..n {
            for j in (i + 1)..n {
                if a[(i, j)].abs() > max {
                    max = a[(i, j)].abs();
                    p = i;
                    q = j;
                }
            }
        }
        if max < 1e-14 {
            break;
        }
        let app = a[(p, p)];
        let aqq = a[(q, q)];
        let apq = a[(p, q)];
        let theta = 0.5 * (aqq - app) / apq;
        let t = theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt());
        let c = 1.0 / (t * t + 1.0).sqrt();
        let s = t * c;
        for k in 0..n {
            let akp = a[(k, p)];
            let akq = a[(k, q)];
            a[(k, p)] = c * akp - s * akq;
            a[(k, q)] = s * akp + c * akq;
        }
        for k in 0..n {
            let apk = a[(p, k)];
            let aqk = a[(q, k)];
            a[(p, k)] = c * apk - s * aqk;
            a[(q, k)] = s * apk + c * aqk;
        }
        a[(p, q)] = 0.0;
        a[(q, p)] = 0.0;
        for k in 0..n {
            let vkp = v[(k, p)];
            let vkq = v[(k, q)];
            v[(k, p)] = c * vkp - s * vkq;
            v[(k, q)] = s * vkp + c * vkq;
        }
    }
    let mut eig = Vec::with_capacity(n);
    for i in 0..n {
        eig.push(a[(i, i)]);
    }
    // sort ascending
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&i, &j| eig[i].partial_cmp(&eig[j]).unwrap());
    let vals: Vec<f64> = idx.iter().map(|&i| eig[i]).collect();
    let mut vecs = Mat::zeros((n, n));
    for (new_j, &old_j) in idx.iter().enumerate() {
        for i in 0..n {
            vecs[(i, new_j)] = v[(i, old_j)];
        }
    }
    (vals, vecs)
}

/// SVD via eigendecomposition of X^T X (m x m, m = samples, small). Returns
/// (U, s, Vt) with singular values descending — used for module eigengenes, Eq. (10).
pub fn svd(x: &Mat) -> (Mat, Vec<f64>, Mat) {
    let (n, m) = (x.nrows(), x.ncols());
    let xt = x.t().to_owned();
    let xtx = xt.dot(x);
    let (evals, v) = jacobi_eigen(&xtx);
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

/// Standardize rows of `x` to mean 0, sd 1 (Methods: x_ik = (y_ik - mean_i)/sd_i).
pub fn standardize_rows(x: &Mat) -> Mat {
    let n = x.nrows();
    let m = x.ncols();
    let mut z = x.clone();
    for i in 0..n {
        let mut s = 0.0;
        for j in 0..m {
            s += x[(i, j)];
        }
        let mean = s / m as f64;
        let mut v = 0.0;
        for j in 0..m {
            let d = x[(i, j)] - mean;
            v += d * d;
        }
        let sd = (v / (m - 1).max(1) as f64).sqrt();
        for j in 0..m {
            z[(i, j)] = if sd > 0.0 { (x[(i, j)] - mean) / sd } else { 0.0 };
        }
    }
    z
}
