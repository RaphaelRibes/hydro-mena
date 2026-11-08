//! Statistics helpers: Pearson r + p-value, chi-square GOF, t CDF, permutation mantel.
use statrs::distribution::{ChiSquared, ContinuousCDF, StudentsT};
use statrs::statistics::Statistics;

pub fn pearson(x: &[f64], y: &[f64]) -> (f64, f64) {
    let n = x.len().min(y.len());
    let sx = &x[..n];
    let sy = &y[..n];
    let mx = sx.mean();
    let my = sy.mean();
    let mut cov = 0.0;
    let mut vx = 0.0;
    let mut vy = 0.0;
    for i in 0..n {
        let dx = sx[i] - mx;
        let dy = sy[i] - my;
        cov += dx * dy;
        vx += dx * dx;
        vy += dy * dy;
    }
    let r = if vx > 0.0 && vy > 0.0 { cov / (vx * vy).sqrt() } else { 0.0 };
    // two-tailed p from t distribution with n-2 df
    let r = r.clamp(-1.0 + 1e-12, 1.0 - 1e-12);
    let df = (n as f64 - 2.0).max(1.0);
    let t = r * (df / (1.0 - r * r)).sqrt();
    let p = if let Ok(dist) = StudentsT::new(0.0, 1.0, df) {
        2.0 * (1.0 - dist.cdf(t.abs()))
    } else {
        f64::NAN
    };
    (r, p)
}

/// Chi-square GOF of observed spacings `d` against Poisson exp(-d) with mean
/// spacing normalized to 1 (paper steps g-i). Returns (chi2, dof, crit at alpha).
pub fn poisson_chi2(d: &[f64], alpha: f64) -> (f64, usize, f64) {
    let u = d.len().saturating_sub(1);
    let chi2 = d.iter().map(|x| (x - 1.0).powi(2)).sum::<f64>(); // E[d]=1
    let crit = ChiSquared::new(u as f64).unwrap().inverse_cdf(1.0 - alpha);
    (chi2, u, crit)
}

pub fn mean(x: &[f64]) -> f64 {
    if x.is_empty() {
        f64::NAN
    } else {
        x.iter().sum::<f64>() / x.len() as f64
    }
}

pub fn std(x: &[f64]) -> f64 {
    if x.len() < 2 {
        0.0
    } else {
        let m = mean(x);
        (x.iter().map(|v| (v - m).powi(2)).sum::<f64>() / (x.len() - 1) as f64).sqrt()
    }
}

/// Mantel test: correlation between two distance vectors, significance by permutation.
pub fn mantel(x: &[f64], y: &[f64], n_perm: usize, rng: &mut rand::rngs::StdRng) -> (f64, f64) {
    let (r_obs, _) = pearson(x, y);
    let n = x.len();
    let mut count = 0usize;
    use rand::seq::SliceRandom;
    use rand::SeedableRng;
    let mut idx: Vec<usize> = (0..n).collect();
    let mut yp = y.to_vec();
    for _ in 0..n_perm {
        idx.shuffle(rng);
        for (i, &j) in idx.iter().enumerate() {
            yp[i] = y[j];
        }
        let (r, _) = pearson(x, &yp);
        if r >= r_obs {
            count += 1;
        }
    }
    let _ = rand::rngs::StdRng::seed_from_u64(0); // keep import used
    (r_obs, (count + 1) as f64 / (n_perm + 1) as f64)
}
