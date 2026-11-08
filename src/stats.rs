//! Statistics helpers: Pearson r + p-value, binned Poisson GOF, KS distance,
//! permutation Mantel test.
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
    // two-tailed p from the t distribution with n-2 df
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

/// Binned chi-square goodness-of-fit of unit-mean spacings `d` against the
/// Poisson law exp(-s) (paper steps j-k).
///
/// The old implementation summed (d-1)^2, which is not a GOF statistic at all:
/// it ignores the shape of the distribution and, on the rank-based spacings the
/// old unfolding produced (all exactly 1.0), was identically zero. Here the
/// spacings are binned, adjacent bins with an expected count < 5 are merged, and
/// the statistic is compared with a chi-square quantile.
///
/// Returns (chi2, dof, crit) with `crit = qchisq(1 - alpha, dof)`.
pub fn poisson_gof(d: &[f64], alpha: f64) -> (f64, usize, f64) {
    if d.is_empty() {
        return (f64::NAN, 0, f64::NAN);
    }
    // bin edges in units of the mean spacing; Poisson(exp(-s)) has P(a<s<b)=e^-a-e^-b
    const EDGES: [f64; 12] = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.3, 1.7, 2.2, 3.0, 4.5, f64::INFINITY];
    let n = d.len() as f64;
    let mut obs: Vec<f64> = vec![0.0; EDGES.len() - 1];
    let mut exp: Vec<f64> = vec![0.0; EDGES.len() - 1];
    for k in 0..EDGES.len() - 1 {
        let (a, b) = (EDGES[k], EDGES[k + 1]);
        exp[k] = n * ((-a).exp() - (-b).exp());
    }
    for &v in d {
        for k in 0..EDGES.len() - 1 {
            if v >= EDGES[k] && v < EDGES[k + 1] {
                obs[k] += 1.0;
                break;
            }
        }
    }
    // merge adjacent bins until every expected count is >= 5
    let mut o: Vec<f64> = Vec::new();
    let mut e: Vec<f64> = Vec::new();
    for k in 0..obs.len() {
        if let (Some(lo), Some(le)) = (o.last_mut(), e.last_mut()) {
            if *le < 5.0 || exp[k] < 5.0 {
                *lo += obs[k];
                *le += exp[k];
                continue;
            }
        }
        o.push(obs[k]);
        e.push(exp[k]);
    }
    let mut chi2 = 0.0;
    for k in 0..o.len() {
        if e[k] > 0.0 {
            chi2 += (o[k] - e[k]).powi(2) / e[k];
        }
    }
    // -1 for the constraint sum(obs) = n, -1 for the estimated mean spacing.
    // It must NOT be floored at 1: with a single merged bin there is no test to
    // perform, and a fabricated dof=1 turns chi2~0 into a spurious "Poisson"
    // verdict. dof=0 means "not estimable" and propagates as such.
    let dof = o.len().saturating_sub(2);
    let crit = ChiSquared::new(dof as f64)
        .map(|c| c.inverse_cdf(1.0 - alpha))
        .unwrap_or(f64::NAN);
    (chi2, dof, crit)
}

/// Kolmogorov-Smirnov distance between the empirical spacing distribution and
/// the Poisson CDF 1-exp(-s). Secondary diagnostic; not used for the decision.
pub fn ks_poisson(d: &[f64]) -> f64 {
    if d.is_empty() {
        return f64::NAN;
    }
    let mut v = d.to_vec();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = v.len() as f64;
    let mut dist = 0.0f64;
    for (i, &s) in v.iter().enumerate() {
        let cdf = 1.0 - (-s).exp();
        let hi = (i + 1) as f64 / n;
        let lo = i as f64 / n;
        dist = dist.max((hi - cdf).abs()).max((cdf - lo).abs());
    }
    dist
}

/// Benjamini-Hochberg q-values. NaN/undefined p-values are excluded from the
/// family (never treated as 1.0): a degenerate test is not evidence of no effect.
/// Returns q aligned to the input, NaN where the input p was NaN.
pub fn bh_qvalues(p: &[f64]) -> Vec<f64> {
    let mut idx: Vec<usize> = (0..p.len()).filter(|&i| p[i].is_finite()).collect();
    let m = idx.len();
    let mut q = vec![f64::NAN; p.len()];
    if m == 0 {
        return q;
    }
    idx.sort_by(|&a, &b| p[a].partial_cmp(&p[b]).unwrap());
    let mut running = 1.0f64;
    for rank in (1..=m).rev() {
        let i = idx[rank - 1];
        let val = (p[i] * m as f64 / rank as f64).min(1.0);
        running = running.min(val);
        q[i] = running;
    }
    q
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

/// Index of the pair (i, j), i < j, in a condensed distance vector of size
/// n*(n-1)/2.
#[inline]
fn condensed_index(n: usize, i: usize, j: usize) -> usize {
    debug_assert!(i < j);
    (i * (2 * n - i - 1)) / 2 + (j - i - 1)
}

/// Mantel test: correlation between two condensed distance matrices, with
/// significance from permuting **node labels** of `y` (the standard null for
/// the Mantel test) rather than shuffling the pair entries, which destroys the
/// matrix structure and inflates significance.
///
/// Returns (r_obs, p, mean r under the null).
pub fn mantel_condensed(
    x: &[f64],
    y: &[f64],
    n: usize,
    n_perm: usize,
    seed: u64,
) -> (f64, f64, f64) {
    use rand::seq::SliceRandom;
    use rand::SeedableRng;
    use rayon::prelude::*;

    let (r_obs, _) = pearson(x, y);
    if n_perm == 0 || n < 3 {
        return (r_obs, f64::NAN, f64::NAN);
    }
    let l = x.len();
    let (count, sum) = (0..n_perm)
        .into_par_iter()
        .fold(
            || (0usize, 0.0f64, vec![0.0f64; l], vec![0usize; n]),
            |(mut c, mut s, mut buf, mut perm), p| {
                let mut rng = rand::rngs::StdRng::seed_from_u64(seed.wrapping_add(p as u64));
                for (i, v) in perm.iter_mut().enumerate() {
                    *v = i;
                }
                perm.shuffle(&mut rng);
                let mut k = 0usize;
                for i in 0..n - 1 {
                    for j in (i + 1)..n {
                        let (a, b) = if perm[i] < perm[j] {
                            (perm[i], perm[j])
                        } else {
                            (perm[j], perm[i])
                        };
                        buf[k] = y[condensed_index(n, a, b)];
                        k += 1;
                    }
                }
                let (r, _) = pearson(x, &buf);
                s += r;
                if r >= r_obs {
                    c += 1;
                }
                (c, s, buf, perm)
            },
        )
        .map(|(c, s, _, _)| (c, s))
        .reduce(|| (0usize, 0.0), |a, b| (a.0 + b.0, a.1 + b.1));
    (
        r_obs,
        (count + 1) as f64 / (n_perm + 1) as f64,
        sum / n_perm as f64,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pearson_perfect_and_anticorrelated() {
        let x: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let y: Vec<f64> = x.iter().map(|v| 2.0 * v + 1.0).collect();
        let (r, p) = pearson(&x, &y);
        assert!((r - 1.0).abs() < 1e-12);
        assert!(p < 1e-6);
    }

    #[test]
    fn poisson_gof_accepts_poisson_samples_rejects_clustered() {
        // exponential spacings = Poisson process inter-arrivals
        use rand::SeedableRng;
        let mut rng = rand::rngs::StdRng::seed_from_u64(7);
        use rand::Rng;
        let mut d: Vec<f64> = (0..2000).map(|_| -rng.gen::<f64>().max(1e-12).ln()).collect();
        let m = d.iter().sum::<f64>() / d.len() as f64;
        for v in d.iter_mut() {
            *v /= m;
        }
        let (chi2, dof, crit) = poisson_gof(&d, 0.01);
        assert!(chi2 <= crit, "exponential spacings rejected: chi2={chi2} crit={crit} dof={dof}");

        // all-equal spacings are maximally non-Poisson
        let flat = vec![1.0f64; 500];
        let (chi2f, _, critf) = poisson_gof(&flat, 0.01);
        assert!(chi2f > critf, "flat spacings must be rejected");
    }

    #[test]
    fn bh_matches_known_example_and_is_monotone() {
        // classic BH illustration; q must be >= p, monotone in rank, capped at 1
        let p = [0.001, 0.008, 0.039, 0.041, 0.042, 0.06, 0.074, 0.205, 0.212, 0.9];
        let q = bh_qvalues(&p);
        for i in 0..p.len() {
            assert!(q[i] >= p[i] - 1e-12, "q < p at {i}");
            assert!(q[i] <= 1.0);
        }
        assert!((q[0] - 0.01).abs() < 1e-9, "q0={}", q[0]); // 0.001*10/1
        // NaN p stays NaN and does not consume a rank
        let q2 = bh_qvalues(&[0.01, f64::NAN, 0.04]);
        assert!(q2[1].is_nan());
        assert!((q2[2] - 0.04).abs() < 1e-9, "q2={}", q2[2]); // 0.04*2/2 = 0.04
    }

    #[test]
    fn mantel_detects_correlation_and_is_null_under_independence() {
        let n = 12usize;
        let mut x = Vec::new();
        let mut y = Vec::new();
        for i in 0..n - 1 {
            for j in (i + 1)..n {
                let d = (i as f64 - j as f64).abs();
                x.push(d);
                y.push(d + 0.01 * (i * j) as f64);
            }
        }
        let (r, p, _) = mantel_condensed(&x, &y, n, 499, 1);
        assert!(r > 0.9, "r={r}");
        assert!(p <= 0.01, "p={p}");
    }
}
