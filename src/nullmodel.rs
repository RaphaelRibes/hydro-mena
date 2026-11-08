//! Configuration-model null models for the network topology indices.
//!
//! A MENA network is compared with "random" graphs in the paper (Deng et al.
//! 2012, Table 1), but a plain Erdos-Renyi null is the wrong null here: it
//! changes the degree distribution as well as the wiring, so a modularity
//! excess could be an artefact of the degrees rather than of the community
//! structure. The configuration model keeps every degree fixed and randomises
//! the wiring, which is the null that isolates modularity and clustering.
//!
//! Graphs are randomised by **edge switching** (the standard configuration-model
//! sampler): pick two distinct edges `(i,j)` and `(k,l)` and propose `(i,l)`
//! and `(k,j)`. Degrees are preserved exactly by construction, and a swap is
//! rejected when it would create a self-loop, a parallel edge, or an edge that
//! already exists. `SWAPS_PER_EDGE * E` proposals per null graph mixes the
//! topology far beyond the mixing time at these sizes.
//!
//! The z-score is `(observed - mean_null) / sd_null`, with the null statistics
//! computed by the **same** functions as the observed ones
//! (`network::greedy_modularity`, `network::clustering_coefficients`,
//! `network::avg_path_length`), so the comparison is like-for-like and no
//! re-implementation can drift from the reported index.
use crate::linalg::Mat;
use crate::network;
use crate::stats;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;
use std::collections::HashSet;

/// Proposals per edge in each null graph. 10 is the usual choice in the
/// ecological network literature; below ~5 the sampler has not mixed.
const SWAPS_PER_EDGE: usize = 10;

/// Attempts allowed before giving up on one proposed swap. Rejection is normal
/// (the graph is sparse), so a cap keeps the loop from stalling on a
/// saturated configuration.
const SWAP_TRIES: usize = 10;

/// Number of null graphs, and the z-scores of the three topology indices the
/// paper reports against the configuration model.
#[derive(Clone, Debug, serde::Serialize)]
pub struct NullReport {
    /// null replicates actually produced (`b` from the call)
    pub b: usize,
    /// z-score of modularity Q against degree-preserving randomisations
    pub modularity_z: f64,
    /// z-score of the average clustering coefficient
    pub avg_cc_z: f64,
    /// z-score of the average path length (GD)
    pub gd_z: f64,
}

/// One null replicate: clone the graph, then perform `SWAPS_PER_EDGE * E`
/// proposed edge switches. Returns the null adjacency as a `Mat`.
fn randomise(adj: &Mat, edges: &[(usize, usize)], seed: u64) -> Mat {
    let mut null = adj.clone();
    let mut present: HashSet<(usize, usize)> = edges.iter().copied().collect();
    let mut e = edges.to_vec();
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let n_swaps = SWAPS_PER_EDGE * e.len().max(1);
    for _ in 0..n_swaps {
        // Both endpoints are drawn fresh on every attempt: picking a pair and
        // then only re-picking one of them would bias the sampler.
        for _try in 0..SWAP_TRIES {
            if e.len() < 2 {
                break;
            }
            let a = rng.gen_range(0..e.len());
            let mut b = rng.gen_range(0..e.len() - 1);
            if b >= a {
                b += 1;
            }
            let (i, j) = e[a];
            let (k, l) = e[b];
            // (i,j),(k,l) -> (i,l),(k,j): degrees preserved for all four ends.
            if i == l || k == j {
                continue;
            }
            let (p, q) = if i < l { (i, l) } else { (l, i) };
            let (r, s) = if k < j { (k, j) } else { (j, k) };
            if p == r && q == s {
                continue;
            }
            if present.contains(&(p, q)) || present.contains(&(r, s)) {
                continue;
            }
            present.remove(&(i.min(j), i.max(j)));
            present.remove(&(k.min(l), k.max(l)));
            present.insert((p, q));
            present.insert((r, s));
            null[(i, j)] = 0.0;
            null[(j, i)] = 0.0;
            null[(k, l)] = 0.0;
            null[(l, k)] = 0.0;
            null[(p, q)] = 1.0;
            null[(q, p)] = 1.0;
            null[(r, s)] = 1.0;
            null[(s, r)] = 1.0;
            e[a] = (p, q);
            e[b] = (r, s);
            break;
        }
    }
    null
}

/// Topology indices of one graph, computed with the same light functions used
/// for the observed graph (no scale-free / hierarchy fits: the null only needs
/// the three compared quantities).
fn topology_stats(a: &Mat) -> (f64, f64, f64) {
    let adj = network::adjacency_list(a);
    let (q, _) = network::greedy_modularity(&adj);
    let cc = network::clustering_coefficients(&adj);
    let avg_cc = stats::mean(&cc);
    let (gd, _) = network::avg_path_length(&adj);
    (q, avg_cc, gd)
}

fn z(observed: f64, samples: &[f64]) -> f64 {
    if samples.len() < 2 || !observed.is_finite() {
        return f64::NAN;
    }
    let m = stats::mean(samples);
    let sd = stats::std(samples);
    // No null variation means the comparison is undefined, not "infinitely
    // significant" -- report that honestly.
    if !(sd > 0.0) || !sd.is_finite() {
        return f64::NAN;
    }
    (observed - m) / sd
}

/// Z-scores of modularity, average clustering and GD against `b`
/// configuration-model randomisations of `adj`.
///
/// Deterministic for a given `(adj, b, seed)`: each replicate gets its own RNG
/// stream (`seed + replicate`), so the result does not depend on the rayon
/// thread count.
pub fn null_zscores(adj: &Mat, b: usize, seed: u64) -> NullReport {
    let n = adj.nrows();
    let mut edges: Vec<(usize, usize)> = Vec::new();
    if b > 0 {
        for i in 0..n {
            for j in (i + 1)..n {
                if adj[(i, j)] > 0.0 {
                    edges.push((i, j));
                }
            }
        }
    }
    if b == 0 || edges.is_empty() {
        return NullReport { b: 0, modularity_z: f64::NAN, avg_cc_z: f64::NAN, gd_z: f64::NAN };
    }
    let (q_obs, cc_obs, gd_obs) = topology_stats(adj);
    let nulls: Vec<(f64, f64, f64)> = (0..b)
        .into_par_iter()
        .map(|r| {
            let null = randomise(adj, &edges, seed.wrapping_add(r as u64));
            topology_stats(&null)
        })
        .collect();
    let qs: Vec<f64> = nulls.iter().map(|t| t.0).collect();
    let ccs: Vec<f64> = nulls.iter().map(|t| t.1).collect();
    let gds: Vec<f64> = nulls.iter().map(|t| t.2).collect();
    NullReport {
        b: nulls.len(),
        modularity_z: z(q_obs, &qs),
        avg_cc_z: z(cc_obs, &ccs),
        gd_z: z(gd_obs, &gds),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn random_graph(n: usize, p: f64, seed: u64) -> Mat {
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        let mut a = Mat::zeros((n, n));
        for i in 0..n {
            for j in (i + 1)..n {
                if rng.gen::<f64>() < p {
                    a[(i, j)] = 1.0;
                    a[(j, i)] = 1.0;
                }
            }
        }
        a
    }

    fn edge_list(a: &Mat) -> Vec<(usize, usize)> {
        let mut e = Vec::new();
        for i in 0..a.nrows() {
            for j in (i + 1)..a.nrows() {
                if a[(i, j)] > 0.0 {
                    e.push((i, j));
                }
            }
        }
        e
    }

    /// Two cliques of `k` nodes with no edges between them: an obvious,
    /// maximal community structure.
    fn two_cliques(k: usize) -> Mat {
        let n = 2 * k;
        let mut a = Mat::zeros((n, n));
        for i in 0..n {
            for j in (i + 1)..n {
                if i / k == j / k {
                    a[(i, j)] = 1.0;
                    a[(j, i)] = 1.0;
                }
            }
        }
        a
    }

    #[test]
    fn null_preserves_every_degree() {
        // the whole point of the configuration model: degrees identical
        let a = random_graph(60, 0.12, 5);
        let mut before: Vec<usize> = (0..a.nrows()).map(|i| (0..a.nrows()).filter(|&j| a[(i, j)] > 0.0).count()).collect();
        before.sort_unstable();
        let e = edge_list(&a);
        for seed in 0..5u64 {
            let null = randomise(&a, &e, seed);
            let mut after: Vec<usize> =
                (0..null.nrows()).map(|i| (0..null.nrows()).filter(|&j| null[(i, j)] > 0.0).count()).collect();
            after.sort_unstable();
            assert_eq!(after, before, "degree sequence changed (seed {seed})");
            // and the wiring actually moved
            assert_ne!(null, a, "null replicate {seed} is the original graph");
        }
    }

    #[test]
    fn random_graph_is_consistent_with_its_own_null() {
        // "Structureless" has to mean *inside the null ensemble*: a graph drawn
        // by the sampler itself must not look significant against further
        // draws from it. A raw Erdos-Renyi graph does not qualify -- ER has
        // E[C] = p by construction while the configuration model on the same
        // (heterogeneous) degree sequence has visibly less clustering and
        // slightly longer paths. At n=70 that is z ~ +5 / -5, a real ensemble
        // difference rather than noise, and it is the reason the null has to
        // hold the degrees fixed: the question is about wiring, not degrees.
        let base = random_graph(70, 0.10, 4242);
        let obs = randomise(&base, &edge_list(&base), 999);
        let r = null_zscores(&obs, 50, 11);
        assert_eq!(r.b, 50);
        for (name, v) in [("modularity", r.modularity_z), ("avg_cc", r.avg_cc_z), ("gd", r.gd_z)] {
            assert!(v.is_finite(), "{name}_z not finite: {r:?}");
            assert!(v.abs() < 4.0, "{name}_z = {v} on a graph from the null ensemble: {r:?}");
        }
    }

    #[test]
    fn planted_communities_beat_the_null() {
        // Two disjoint cliques: modularity is 0.5 by construction and no
        // degree-preserving randomisation reaches it, so z must be large.
        let a = two_cliques(25);
        let r = null_zscores(&a, 40, 7);
        assert!(r.modularity_z > 3.0, "modularity_z = {}: {r:?}", r.modularity_z);
        assert!(r.avg_cc_z > 3.0, "avg_cc_z = {}: {r:?}", r.avg_cc_z);
    }

    #[test]
    fn degenerate_inputs_are_not_estimable() {
        let a = random_graph(30, 0.1, 1);
        assert!(null_zscores(&a, 0, 1).modularity_z.is_nan());
        let empty = Mat::zeros((30, 30));
        let r = null_zscores(&empty, 10, 1);
        assert_eq!(r.b, 0);
        assert!(r.modularity_z.is_nan());
    }
}
