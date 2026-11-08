//! Trait relationships (paper: Mantel tests + trait-based OTU significance).
//! Mantel: Pearson correlation between upper-triangular connectivity-distance
//! |k_i - k_j| and GS-distance sqrt(sum_h (GS_ih - GS_jh)^2), permuted.
use crate::eigengene::trait_significance;
use crate::linalg::Mat;
use crate::network::adjacency_list;
use crate::stats;

use rand::rngs::StdRng;
use rand::SeedableRng;

pub struct MantelResult {
    pub r: f64,
    pub p: f64,
    pub n_perm: usize,
}

pub fn mantel_connectivity_vs_traits(A: &Mat, z: &Mat, traits: &Mat, n_perm: usize, seed: u64) -> MantelResult {
    let mut rng = StdRng::seed_from_u64(seed);
    let adj = adjacency_list(A);
    let n = adj.len();
    let k: Vec<f64> = adj.iter().map(|a| a.len() as f64).collect();
    let gs = trait_significance(z, traits);
    let t = traits.ncols();
    // upper-triangle distances
    let mut kd = Vec::new();
    let mut gd = Vec::new();
    for i in 0..n {
        for j in (i + 1)..n {
            kd.push((k[i] - k[j]).abs());
            let mut s = 0.0;
            for h in 0..t {
                let d = gs[(i, h)] - gs[(j, h)];
                s += d * d;
            }
            gd.push(s.sqrt());
        }
    }
    let (r, p) = stats::mantel(&kd, &gd, n_perm, &mut rng);
    MantelResult { r, p, n_perm }
}
