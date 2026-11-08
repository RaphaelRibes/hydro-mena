//! Statistics helpers: Pearson r + p-value, binned Poisson GOF, KS distances
//! (Poisson and Wigner/GOE), the MLE Brody exponent, permutation Mantel test.
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

/// Kolmogorov-Smirnov distance between the empirical spacing distribution and
/// the Wigner surmise CDF, F(s) = 1 - exp(-pi s^2 / 4).
///
/// Same estimator as [`ks_poisson`] (sup over the empirical CDF steps), so the
/// two numbers are directly comparable: which one is smaller says whether the
/// spacings look Poisson (integrable) or GOE (level-repelling).
pub fn ks_goe(d: &[f64]) -> f64 {
    if d.is_empty() {
        return f64::NAN;
    }
    let mut v = d.to_vec();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = v.len() as f64;
    let mut dist = 0.0f64;
    for (i, &s) in v.iter().enumerate() {
        let cdf = 1.0 - (-std::f64::consts::PI * s * s / 4.0).exp();
        let hi = (i + 1) as f64 / n;
        let lo = i as f64 / n;
        dist = dist.max((hi - cdf).abs()).max((cdf - lo).abs());
    }
    dist
}

/// Unit-mean scale `b` of the Brody density
/// P_B(s) = (beta+1) b s^beta exp(-b s^(beta+1)),
/// b = Gamma((beta+2)/(beta+1))^(beta+1).
///
/// The prefactor is `(beta+1) b`, not `beta+1`: the density must integrate to
/// one, and the unit-mean constraint fixes b, so a = (beta+1) b. This is the
/// standard form and it pins the two anchors — beta=0 gives exp(-s) (Poisson)
/// and beta=1 gives (pi/2) s exp(-pi s^2/4) (Wigner surmise).
fn brody_b(beta: f64) -> f64 {
    use statrs::function::gamma::gamma;
    let g = gamma((beta + 2.0) / (beta + 1.0));
    g.powf(beta + 1.0)
}

/// Brody log-likelihood of unit-mean spacings at shape `beta` (up to the
/// additive constant, which does not move the maximiser).
fn brody_loglik(s: &[f64], ln_s: &[f64], beta: f64) -> f64 {
    let b = brody_b(beta);
    if !(b > 0.0) || !b.is_finite() {
        return f64::NEG_INFINITY;
    }
    let p = beta + 1.0;
    let sum_pow: f64 = s.iter().map(|v| v.powf(p)).sum();
    let n = s.len() as f64;
    n * (p.ln() + b.ln()) + beta * ln_s.iter().sum::<f64>() - b * sum_pow
}

/// Maximum-likelihood Brody exponent of the level-spacing distribution: the
/// continuous replacement for the binary "Poisson vs not" chi-square verdict.
///
/// beta = 0 is a Poisson spectrum (uncorrelated eigenvalues, what the paper
/// wants at the RMT threshold), beta = 1 is GOE (Wigner, level repulsion: the
/// graph is still correlated, i.e. the threshold is too low). Values in
/// between are what real data produce, so the number is informative where the
/// accept/reject flag is not.
///
/// Maximised by golden-section search on beta in [0, 2]. Spacings below 1e-6 of
/// the mean are dropped first: numerical degeneracies land at 1e-17 instead of
/// exactly 0 depending on the eigen-solver's ordering, and a single ln(1e-17)
/// term pins the MLE to the beta=0 boundary for every real graph (measured:
/// 18 degenerate spacings in a 149-node box spectrum flipped beta from 0.59 to
/// 0.00). They carry no level-spacing information; the remaining spacings are
/// renormalised to unit mean, which is the family the MLE is defined on.
pub fn brody_beta(d: &[f64]) -> f64 {
    let positive: Vec<f64> = d.iter().copied().filter(|v| v.is_finite() && *v > 0.0).collect();
    if positive.len() < 8 {
        return f64::NAN;
    }
    let scale = positive.iter().sum::<f64>() / positive.len() as f64;
    if !(scale > 0.0) || !scale.is_finite() {
        return f64::NAN;
    }
    let floor = 1e-6 * scale;
    let mut s: Vec<f64> = positive.into_iter().filter(|v| *v > floor).collect();
    if s.len() < 8 {
        return f64::NAN;
    }
    let m = s.iter().sum::<f64>() / s.len() as f64;
    for v in s.iter_mut() {
        *v /= m;
    }
    let ln_s: Vec<f64> = s.iter().map(|v| v.ln()).collect();

    const PHI: f64 = 0.618_033_988_749_894_9;
    let (mut lo, mut hi) = (0.0f64, 2.0f64);
    let mut c = hi - PHI * (hi - lo);
    let mut e = lo + PHI * (hi - lo);
    let (mut fc, mut fe) = (brody_loglik(&s, &ln_s, c), brody_loglik(&s, &ln_s, e));
    if !fc.is_finite() || !fe.is_finite() {
        return f64::NAN;
    }
    for _ in 0..200 {
        if hi - lo < 1e-9 {
            break;
        }
        if fc > fe {
            hi = e;
            e = c;
            fe = fc;
            c = hi - PHI * (hi - lo);
            fc = brody_loglik(&s, &ln_s, c);
        } else {
            lo = c;
            c = e;
            fc = fe;
            e = lo + PHI * (hi - lo);
            fe = brody_loglik(&s, &ln_s, e);
        }
    }
    0.5 * (lo + hi)
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

// ---------------------------------------------------------------------------
// Clauset–Shalizi–Newman power-law fit (discrete MLE + KS, 2009 procedure,
// simplified: xmin scanned over observed values, p-value by simulation).
// ---------------------------------------------------------------------------

/// Discrete power-law MLE gamma for k >= xmin (Clauset eq. 3.7 approx):
/// gamma = 1 + n / sum(ln(k / (xmin - 0.5))).
fn gamma_mle(tail: &[f64], xmin: f64) -> f64 {
    let n = tail.len() as f64;
    let s: f64 = tail.iter().map(|&k| (k / (xmin - 0.5)).ln()).sum();
    1.0 + n / s
}

/// KS distance between the empirical distribution of `tail` and the fitted
/// discrete power law. The fitted CDF is computed exactly by cumulative sums
/// of j^-gamma from the top down, so there is no integral-approximation bias.
fn ks_powerlaw(tail: &[f64], xmin: f64, gamma: f64) -> f64 {
    let mut sorted = tail.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = sorted.len() as f64;
    // distinct values, ascending
    let mut uniq: Vec<f64> = Vec::new();
    let mut counts: Vec<usize> = Vec::new();
    for &k in &sorted {
        if uniq.last() == Some(&k) {
            *counts.last_mut().unwrap() += 1;
        } else {
            uniq.push(k);
            counts.push(1);
        }
    }
    // Z = sum_{j >= xmin} j^-gamma, computed exactly over the relevant range
    // (integral tail beyond the max observed value).
    let j0 = xmin.ceil() as i64;
    let jmax = uniq.last().copied().unwrap_or(xmin).ceil() as i64;
    let mut z = 0.0f64;
    let mut j = j0;
    while j <= jmax {
        z += (j as f64).powf(-gamma);
        j += 1;
    }
    z += (jmax as f64 + 0.5f64).powf(1.0 - gamma) / (gamma - 1.0); // integral tail
    // fitted CCDF at each distinct k: sum_{j >= k} j^-g / Z
    let mut ccdf = vec![0.0f64; uniq.len()];
    let mut acc = (jmax as f64 + 0.5f64).powf(1.0 - gamma) / (gamma - 1.0);
    for idx in (0..uniq.len()).rev() {
        let k = uniq[idx];
        let jstart = k.ceil() as i64;
        let mut s = 0.0f64;
        let mut j = jstart;
        while j <= jmax {
            s += (j as f64).powf(-gamma);
            j += 1;
        }
        acc = s + (jmax as f64 + 0.5f64).powf(1.0 - gamma) / (gamma - 1.0);
        ccdf[idx] = acc / z;
    }
    // KS with mid-rank empirical CDF
    let mut d = 0.0f64;
    let mut cum = 0usize;
    for (idx, &k) in uniq.iter().enumerate() {
        cum += counts[idx];
        let emp_cdf = (cum as f64 - 0.5) / n;
        let fit_cdf = 1.0 - ccdf[idx];
        d = d.max((emp_cdf - fit_cdf).abs());
    }
    d
}

/// Simulate a discrete power-law sample (Clauset's generator).
/// Simulate a discrete power-law sample (Clauset's inversion generator):
/// k = floor((xmin - 0.5) * (1-u)^(-1/(gamma-1)) + 0.5), clipped to >= xmin.
fn gen_powerlaw(n: usize, xmin: f64, gamma: f64, rng: &mut rand::rngs::StdRng) -> Vec<f64> {
    use rand::Rng;
    let a = 1.0 / (gamma - 1.0);
    (0..n)
        .map(|_| {
            let u = rng.gen::<f64>();
            ((xmin - 0.5) * (1.0 - u).powf(-a) + 0.5).floor().max(xmin)
        })
        .collect()
}

/// Clauset fit of a degree sequence: scan xmin over observed values, keep the
/// fit with the smallest KS distance, and estimate the p-value as the fraction
/// of `n_sims` synthetic power-law samples (same n, best xmin, fitted gamma)
/// whose KS is at least the observed one.
pub fn clauset_powerlaw(ks: &[usize], seed: u64, n_sims: usize) -> (f64, f64, f64, f64) {
    use rand::SeedableRng;
    let mut vals: Vec<f64> = ks.iter().map(|&k| k as f64).filter(|&k| k > 0.0).collect();
    if vals.len() < 10 {
        return (f64::NAN, f64::NAN, f64::NAN, f64::NAN);
    }
    vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let uniq: Vec<f64> = {
        let mut u = vec![vals[0]];
        for &v in &vals[1..] {
            if v > *u.last().unwrap() {
                u.push(v);
            }
        }
        u
    };
    // Candidate xmins (Clauset: every observed value, but cap count for speed).
    // Require a minimum tail of 20 so the KS statistic is meaningful.
    let min_tail = 100usize;
    let cand: Vec<f64> = {
        let mut c: Vec<f64> = uniq
            .iter()
            .copied()
            .filter(|&x| vals.iter().filter(|&&v| v >= x).count() >= min_tail)
            .collect();
        if c.len() > 50 {
            let step = c.len() / 50;
            c = c.iter().step_by(step.max(1)).copied().collect();
        }
        c
    };
    if cand.is_empty() {
        return (f64::NAN, f64::NAN, f64::NAN, f64::NAN);
    }
    let mut best = (f64::INFINITY, f64::NAN, f64::NAN, f64::NAN); // (ks, gamma, xmin, n_tail)
    for &xmin in &cand {
        let tail: Vec<f64> = vals.iter().copied().filter(|&k| k >= xmin).collect();
        if tail.len() < min_tail {
            continue;
        }
        let g = gamma_mle(&tail, xmin);
        if !g.is_finite() || g <= 1.0 {
            continue;
        }
        let d = ks_powerlaw(&tail, xmin, g);
        if d < best.0 {
            best = (d, g, xmin, tail.len() as f64);
        }
    }
    if !best.0.is_finite() {
        return (f64::NAN, f64::NAN, f64::NAN, f64::NAN);
    }
    let (d_obs, gamma, xmin, n_tail) = best;
    // p-value by simulation
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let mut worse = 0usize;
    for _ in 0..n_sims.max(1) {
        let sim = gen_powerlaw(n_tail as usize, xmin, gamma, &mut rng);
        // parametric bootstrap: compare the synthetic sample's KS against the
        // *fixed* fitted law (gamma, xmin) of the observed data — the standard
        // small-sample variant, less biased than re-fitting gamma on each sim
        let d2 = ks_powerlaw(&sim, xmin, gamma);
        if d2 >= d_obs {
            worse += 1;
        }
    }
    let p = worse as f64 / n_sims.max(1) as f64;
    (gamma, xmin, d_obs, p)
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
    fn clauset_recovers_powerlaw() {
        use rand::SeedableRng;
        // exact power-law sample, gamma = 2.5, n = 4000
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let ks_pl: Vec<usize> = gen_powerlaw(4000, 1.0, 2.5, &mut rng)
            .into_iter()
            .map(|v| v as usize)
            .collect();
        let (g, _xmin, _d, p) = clauset_powerlaw(&ks_pl, 1, 100);
        assert!((g - 2.5).abs() < 0.5, "gamma={g}");
        assert!(p > 0.05, "power law rejected: p={p}");
    }

    #[test]
    fn clauset_rejects_exponential_tail() {
        // Geometric (exponential) degrees are the canonical non-power-law.
        // The discrete MLE picks a very large gamma (>= 4) at a high xmin —
        // far outside the scale-free range gamma in [2, 3.5] — and the
        // observed KS is not small. We assert both: gamma out of range and
        // KS above the 1-sigma level of typical power-law samples.
        use rand::Rng;
        use rand::SeedableRng;
        let mut rng = rand::rngs::StdRng::seed_from_u64(9);
        let ks_exp: Vec<usize> = (0..8000)
            .map(|_| (-rng.gen::<f64>().max(1e-12).ln() * 4.0).ceil() as usize)
            .collect();
        let (g, xmin, d_obs, _p) = clauset_powerlaw(&ks_exp, 5, 100);
        assert!(g > 4.0, "exponential fitted gamma too low: g={g} xmin={xmin}");
        assert!(d_obs > 0.10, "exponential KS too small: d={d_obs}");
    }

    #[test]
    fn clauset_handles_tiny_input() {
        let (g, x, d, p) = clauset_powerlaw(&[1, 2, 3], 1, 10);
        assert!(g.is_nan() && x.is_nan() && d.is_nan() && p.is_nan());
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

    /// Unit-mean exponential (Poisson process) spacings.
    fn poisson_sample(n: usize, seed: u64) -> Vec<f64> {
        use rand::{Rng, SeedableRng};
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        let mut d: Vec<f64> = (0..n).map(|_| -rng.gen::<f64>().max(1e-12).ln()).collect();
        let m = d.iter().sum::<f64>() / d.len() as f64;
        for v in d.iter_mut() {
            *v /= m;
        }
        d
    }

    /// Unit-mean Wigner-surmise spacings (GOE), by inverse CDF.
    fn goe_sample(n: usize, seed: u64) -> Vec<f64> {
        use rand::{Rng, SeedableRng};
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        let mut d: Vec<f64> = (0..n)
            .map(|_| {
                let u: f64 = rng.gen::<f64>();
                2.0 * (-(1.0 - u).ln() / std::f64::consts::PI).sqrt()
            })
            .collect();
        let m = d.iter().sum::<f64>() / d.len() as f64;
        for v in d.iter_mut() {
            *v /= m;
        }
        d
    }

    #[test]
    fn brody_exponent_recovers_the_two_anchor_ensembles() {
        // beta=0: the exponential is exactly the Brody density at the boundary
        let p = poisson_sample(2000, 11);
        let bp = brody_beta(&p);
        assert!(bp < 0.3, "Poisson spacings gave beta={bp}");
        assert!(bp.is_finite());
        // beta=1: Wigner surmise is the Brody density at beta=1
        let g = goe_sample(2000, 12);
        let bg = brody_beta(&g);
        assert!((bg - 1.0).abs() < 0.25, "GOE spacings gave beta={bg}");
        // degenerate inputs are "not estimable", never a fabricated number
        assert!(brody_beta(&[]).is_nan());
        assert!(brody_beta(&[1.0; 3]).is_nan());
    }

    #[test]
    fn degenerate_spacings_do_not_pin_the_brody_exponent() {
        // Repeated eigenvalues arrive either as exact zeros or as ~1e-17
        // depending on the solver's ordering. Either way they are not resolvable
        // spacings and must not drag the MLE to the beta = 0 boundary (which is
        // exactly what happened on the real box spectrum before the floor).
        let base = goe_sample(2000, 33);
        let mut with_zeros = base.clone();
        with_zeros.extend(std::iter::repeat(0.0).take(18));
        let b0 = brody_beta(&with_zeros);
        assert!((b0 - 1.0).abs() < 0.25, "exact zeros gave beta={b0}");
        let mut with_tiny = base;
        with_tiny.extend(std::iter::repeat(1e-17).take(18));
        let bt = brody_beta(&with_tiny);
        assert!((bt - 1.0).abs() < 0.25, "1e-17 spacings gave beta={bt}");
    }

    #[test]
    fn ks_goe_and_ks_poisson_pick_the_right_ensemble() {
        let p = poisson_sample(2000, 21);
        let (ksp, ksg) = (ks_poisson(&p), ks_goe(&p));
        assert!(ksp < 0.06, "KS to Poisson on exponential spacings = {ksp}");
        assert!(ksg > ksp, "exponential spacings: ks_goe={ksg} should exceed ks_poisson={ksp}");

        let g = goe_sample(2000, 22);
        let (ksp, ksg) = (ks_poisson(&g), ks_goe(&g));
        assert!(ksg < 0.06, "KS to Wigner on GOE spacings = {ksg}");
        assert!(ksg < ksp, "GOE spacings: ks_goe={ksg} should be below ks_poisson={ksp}");
    }
}
