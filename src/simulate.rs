//! Synthetic data generator: planted co-abundance modules whose eigengene
//! profiles track environmental traits, so the Fig-5 heatmap has real signal.
//! Output: relative abundance matrix (OTUs x samples, columns sum to 1) and a
//! sample x trait matrix.
use crate::linalg::Mat;
use rand::rngs::StdRng;

use rand::{Rng, SeedableRng};
use rand_distr::{Distribution, Normal};

pub struct SimData {
    pub abundance: Mat, // n_otus x n_samples, relative abundances
    pub traits: Mat,    // n_samples x n_traits
    pub otu_names: Vec<String>,
    pub sample_names: Vec<String>,
    pub trait_names: Vec<String>,
    pub module_of: Vec<usize>, // planted module per OTU (background = n_modules)
}

const DEFAULT_TRAITS: [&str; 4] = ["O2", "Alkane_degradation", "PAH_degradation", "pH"];

pub fn simulate(
    n_otus: usize,
    n_samples: usize,
    n_modules: usize,
    n_traits: usize,
    n_background: usize,
    noise: f64,
    seed: u64,
) -> SimData {
    let mut rng = StdRng::seed_from_u64(seed);
    let normal = Normal::new(0.0, 1.0).unwrap();
    let trait_names: Vec<String> = (0..n_traits)
        .map(|i| {
            DEFAULT_TRAITS
                .get(i)
                .map(|s| s.to_string())
                .unwrap_or_else(|| format!("trait_{i}"))
        })
        .collect();

    // smooth environmental trajectories across ordered samples
    let mut T = Mat::zeros((n_samples, n_traits));
    for h in 0..n_traits {
        for i in 0..n_samples {
            T[(i, h)] = normal.sample(&mut rng);
        }
        // 3-point moving average
        for i in 1..n_samples - 1 {
            T[(i, h)] = (T[(i - 1, h)] + T[(i, h)] + T[(i + 1, h)]) / 3.0;
        }
    }

    // module eigengene profiles; module 0 tracks trait 0, module 2 tracks trait 1,
    // module 1 anti-tracks trait 2 (if present)
    let mut E = Mat::zeros((n_modules, n_samples));
    for b in 0..n_modules {
        for i in 0..n_samples {
            E[(b, i)] = normal.sample(&mut rng);
        }
        for i in 1..n_samples - 1 {
            E[(b, i)] = (E[(b, i - 1)] + E[(b, i)] + E[(b, i + 1)]) / 3.0;
        }
    }
    if n_modules >= 1 && n_traits >= 1 {
        for i in 0..n_samples {
            E[(0, i)] = T[(i, 0)] + 0.25 * normal.sample(&mut rng);
        }
    }
    if n_modules >= 3 && n_traits >= 2 {
        for i in 0..n_samples {
            E[(2, i)] = T[(i, 1)] + 0.25 * normal.sample(&mut rng);
        }
    }
    if n_modules >= 2 && n_traits >= 3 {
        for i in 0..n_samples {
            E[(1, i)] = -0.8 * T[(i, 2)] + 0.3 * normal.sample(&mut rng);
        }
    }

    let n_planted = n_otus.saturating_sub(n_background);
    let mut assign: Vec<usize> = (0..n_planted).map(|i| i % n_modules).collect();
    use rand::seq::SliceRandom;
    assign.shuffle(&mut rng);
    let mut module_of: Vec<usize> = vec![n_modules; n_otus];
    for i in 0..n_planted {
        module_of[i] = assign[i];
    }

    let mut X = Mat::zeros((n_otus, n_samples));
    for i in 0..n_planted {
        let a = rng.gen_range(0.6..1.0) * if rng.gen_bool(0.5) { -1.0 } else { 1.0 };
        for j in 0..n_samples {
            X[(i, j)] = a * E[(assign[i], j)] + noise * normal.sample(&mut rng);
        }
    }
    for i in n_planted..n_otus {
        for j in 0..n_samples {
            X[(i, j)] = normal.sample(&mut rng);
        }
    }

    // to relative abundances per sample
    let mut Y = Mat::zeros((n_otus, n_samples));
    for j in 0..n_samples {
        let mut col: Vec<f64> = (0..n_otus).map(|i| (1.5 * X[(i, j)]).exp()).collect();
        let s: f64 = col.iter().sum();
        for i in 0..n_otus {
            Y[(i, j)] = col[i] / s;
        }
        col.clear();
    }

    SimData {
        abundance: Y,
        traits: T,
        otu_names: (0..n_otus).map(|i| format!("OTU_{i:04}")).collect(),
        sample_names: (0..n_samples).map(|i| format!("S{}", i + 1)).collect(),
        trait_names,
        module_of,
    }
}
