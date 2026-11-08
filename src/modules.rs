//! Module roles (Eq. 8-9): within-module connectivity zi, participation
//! coefficient Pi, and the 4-category classification (peripheral / connector /
//! module hub / network hub with zi<=2.5, Pi<=0.62 thresholds).
//!
//! Module detection itself is selected by [`ModulesAlgo`]: full multilevel
//! Louvain (default) or the previous single-level greedy local moving.
use crate::linalg::Mat;
use crate::network::{adjacency_list, greedy_modularity, louvain_adj};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use serde::Serialize;
use std::collections::HashMap;
use std::str::FromStr;

/// Which modularity optimiser [`detect_modules`] runs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ModulesAlgo {
    /// Full multilevel Louvain (default): phase 1 local moving + phase 2
    /// aggregation, repeated on the coarsened graph.
    Louvain,
    /// Single-level greedy local moving — the previous behaviour, kept for
    /// ablations.
    Greedy,
}

impl Default for ModulesAlgo {
    fn default() -> Self {
        ModulesAlgo::Louvain
    }
}

impl ModulesAlgo {
    pub fn as_str(self) -> &'static str {
        match self {
            ModulesAlgo::Louvain => "louvain",
            ModulesAlgo::Greedy => "greedy",
        }
    }
}

impl FromStr for ModulesAlgo {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_ascii_lowercase().as_str() {
            "louvain" => Ok(ModulesAlgo::Louvain),
            "greedy" => Ok(ModulesAlgo::Greedy),
            other => Err(format!(
                "unknown module algorithm `{other}` (expected louvain or greedy)"
            )),
        }
    }
}

impl std::fmt::Display for ModulesAlgo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

#[derive(Clone, Debug)]
pub struct Role {
    pub node: usize,
    pub module: usize,
    pub zi: f64,
    pub pi: f64,
    pub connectivity: f64,
    pub role: &'static str,
}

pub struct ModuleAnalysis {
    pub modularity: f64,
    pub n_modules: usize,
    pub members: Vec<Vec<usize>>, // module -> node indices, sorted by size desc
    pub module_of: Vec<usize>,
    pub roles: Vec<Role>,
}

/// Bootstrap stability of a module partition (roadmap item 4).
///
/// Samples are resampled with replacement, the network is rebuilt at a fixed
/// similarity threshold, modules are re-detected with the same optimiser, and
/// every reference module is scored by its best Jaccard overlap with any
/// bootstrap module. `mean_jaccard` near 1 means the partition is stable under
/// resampling; near 0 it is a property of this particular sample draw.
#[derive(Clone, Debug, Serialize)]
pub struct BootReport {
    pub n_boot: usize,
    pub mean_jaccard: f64,
    /// mean Jaccard of each reference module (same order as its partition)
    pub per_module_jaccard: Vec<f64>,
}

/// Single entry point for module detection on an adjacency matrix: `algo`
/// selects the modularity optimiser, everything downstream (membership, roles,
/// zi, Pi, Q) is identical. Louvain uses a fixed seed so runs are reproducible.
pub fn detect_modules(a: &Mat, algo: ModulesAlgo) -> ModuleAnalysis {
    detect_modules_adj(&adjacency_list(a), algo)
}

/// Same as [`detect_modules`] but on an already-built adjacency list, so the
/// pipeline can reuse the list it built for `topology_parts_with`.
pub fn detect_modules_adj(adj: &[Vec<usize>], algo: ModulesAlgo) -> ModuleAnalysis {
    let (m, assign) = match algo {
        ModulesAlgo::Louvain => louvain_adj(adj, 42),
        ModulesAlgo::Greedy => greedy_modularity(adj),
    };
    analyze_modules_with(adj, m, assign)
}

pub fn analyze_modules(a: &Mat) -> ModuleAnalysis {
    detect_modules(a, ModulesAlgo::default())
}

/// Module roles on an existing partition: the caller already optimised
/// modularity (see `network::topology_parts`), so it must not be run twice.
pub fn analyze_modules_with(adj: &[Vec<usize>], m: f64, mut assign: Vec<usize>) -> ModuleAnalysis {
    let n = adj.len();
    // order modules by size desc and relabel 0..K-1
    let mut sizes: HashMap<usize, usize> = HashMap::new();
    for &a in &assign {
        *sizes.entry(a).or_insert(0) += 1;
    }
    let mut order: Vec<usize> = sizes.keys().cloned().collect();
    // Tie-break equal sizes on the old module id: HashMap iteration order is
    // randomized per process, and otherwise two same-size modules could swap
    // labels between identical runs.
    order.sort_by_key(|c| (std::cmp::Reverse(sizes[c]), *c));
    let mut remap = HashMap::new();
    for (new, old) in order.iter().enumerate() {
        remap.insert(*old, new);
    }
    for a in assign.iter_mut() {
        *a = remap[a];
    }
    let k_modules = order.len();
    let k: Vec<f64> = adj.iter().map(|a| a.len() as f64).collect();

    // within-module degree kib per node
    let mut kib: Vec<HashMap<usize, f64>> = vec![HashMap::new(); n];
    for i in 0..n {
        for &j in &adj[i] {
            if assign[j] == assign[i] {
                *kib[i].entry(assign[i]).or_insert(0.0) += 1.0;
            } else {
                *kib[i].entry(assign[j]).or_insert(0.0) += 1.0;
            }
        }
    }

    // mean and sd of within-module connectivity per module (Eq. 8)
    let mut mean_kb = vec![0.0; k_modules];
    let mut sd_kb = vec![0.0; k_modules];
    for b in 0..k_modules {
        let vals: Vec<f64> = (0..n)
            .filter(|&i| assign[i] == b)
            .map(|i| *kib[i].get(&b).unwrap_or(&0.0))
            .collect();
        let m = crate::stats::mean(&vals);
        let s = crate::stats::std(&vals);
        mean_kb[b] = m;
        sd_kb[b] = s;
    }

    let mut roles = Vec::with_capacity(n);
    for i in 0..n {
        let b = assign[i];
        let ki = k[i];
        let z = if sd_kb[b] > 0.0 {
            (kib[i].get(&b).copied().unwrap_or(0.0) - mean_kb[b]) / sd_kb[b]
        } else {
            0.0
        };
        let pi = if ki > 0.0 {
            1.0 - kib[i].values().map(|v| (v / ki).powi(2)).sum::<f64>()
        } else {
            0.0
        };
        let role = if z <= 2.5 && pi <= 0.62 {
            "peripheral"
        } else if z <= 2.5 && pi > 0.62 {
            "connector"
        } else if z > 2.5 && pi <= 0.62 {
            "module hub"
        } else {
            "network hub"
        };
        roles.push(Role {
            node: i,
            module: b,
            zi: z,
            pi,
            connectivity: ki,
            role,
        });
    }

    let mut members: Vec<Vec<usize>> = vec![Vec::new(); k_modules];
    for i in 0..n {
        members[assign[i]].push(i);
    }
    for m in members.iter_mut() {
        m.sort();
    }

    ModuleAnalysis {
        modularity: m,
        n_modules: k_modules,
        members,
        module_of: assign,
        roles,
    }
}

pub fn modularity_of(A: &Mat) -> f64 {
    let adj = adjacency_list(A);
    louvain_adj(&adj, 42).0
}

/// Bootstrap module stability: resample `n_boot` sample columns with
/// replacement from the pool `samples`, rebuild the graph at the fixed
/// threshold `st`, re-detect modules with `algo`, and return how well the
/// reference partition (the one computed from the full table at the same `st`)
/// survives.
///
/// `abundance` is OTUs x samples and must already carry the pipeline's
/// preprocessing (`--clr`); `samples` holds the column indices to draw from
/// (pass `0..ncols` for a plain bootstrap over all samples). The reference
/// partition is restricted to nodes with at least one edge at `st`, exactly
/// like the pipeline's reported partition, and replicates drop isolated nodes
/// the same way. A reference module that a replicate splits or loses simply
/// scores a low Jaccard — no replicate is silently discarded.
pub fn bootstrap_stability(
    abundance: &Mat,
    samples: &[usize],
    st: f64,
    algo: ModulesAlgo,
    n_boot: usize,
    seed: u64,
) -> BootReport {
    let n = abundance.nrows();
    let pool: Vec<usize> = if samples.is_empty() {
        (0..abundance.ncols()).collect()
    } else {
        samples.to_vec()
    };
    let m = pool.len();

    // Reference partition on the full table at the same threshold/optimiser:
    // stability is measured against the partition the pipeline reports, not
    // against a re-labelled or re-tuned one.
    let (_, a_ref) = crate::pipeline::adjacency_at(abundance, st);
    let reference = members_on(&a_ref, &crate::rmt::kept_nodes(&a_ref), algo);
    let k = reference.len();
    if n_boot == 0 || k == 0 || n == 0 || m < 2 {
        return BootReport {
            n_boot: 0,
            mean_jaccard: f64::NAN,
            per_module_jaccard: Vec::new(),
        };
    }

    let mut rng = StdRng::seed_from_u64(seed);
    let mut per = vec![0.0f64; k];
    // one scratch table, overwritten per replicate (never stored: n_boot
    // copies of the abundance matrix would dominate memory on real tables)
    let mut x = Mat::zeros((n, m));
    for _ in 0..n_boot {
        for dst in 0..m {
            let src = pool[rng.gen_range(0..m)];
            for i in 0..n {
                x[(i, dst)] = abundance[(i, src)];
            }
        }
        let (_, a) = crate::pipeline::adjacency_at(&x, st);
        let boot_mods = members_on(&a, &crate::rmt::kept_nodes(&a), algo);
        for (b, rset) in reference.iter().enumerate() {
            // Best-overlap mapping to the reference module: the replicate
            // module with the largest Jaccard is this module's counterpart.
            let best = boot_mods
                .iter()
                .map(|bset| jaccard(rset, bset))
                .fold(0.0f64, f64::max);
            per[b] += best;
        }
    }
    for v in per.iter_mut() {
        *v /= n_boot as f64;
    }
    let mean_jaccard = per.iter().sum::<f64>() / k as f64;
    BootReport {
        n_boot,
        mean_jaccard,
        per_module_jaccard: per,
    }
}

/// Modules of the graph on `a` restricted to `kept`, with members as original
/// node indices (ascending) so overlaps with the reference partition compare
/// like with like.
fn members_on(a: &Mat, kept: &[usize], algo: ModulesAlgo) -> Vec<Vec<usize>> {
    if kept.is_empty() {
        return Vec::new();
    }
    let p = kept.len();
    let sub = Mat::from_shape_fn((p, p), |(x, y)| a[(kept[x], kept[y])]);
    detect_modules(&sub, algo)
        .members
        .iter()
        .map(|mem| {
            let mut v: Vec<usize> = mem.iter().map(|&i| kept[i]).collect();
            v.sort_unstable();
            v
        })
        .collect()
}

/// Jaccard index of two ascending node lists.
fn jaccard(a: &[usize], b: &[usize]) -> f64 {
    let (mut i, mut j, mut inter) = (0usize, 0usize, 0usize);
    while i < a.len() && j < b.len() {
        match a[i].cmp(&b[j]) {
            std::cmp::Ordering::Less => i += 1,
            std::cmp::Ordering::Greater => j += 1,
            std::cmp::Ordering::Equal => {
                inter += 1;
                i += 1;
                j += 1;
            }
        }
    }
    let union = a.len() + b.len() - inter;
    if union == 0 {
        0.0
    } else {
        inter as f64 / union as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detect_modules_dispatches_and_keeps_the_report_shape() {
        // two disjoint triangles: both optimisers must recover 2 modules
        let mut a = Mat::zeros((6, 6));
        for (i, j) in [(0, 1), (0, 2), (1, 2), (3, 4), (3, 5), (4, 5)] {
            a[(i, j)] = 1.0;
            a[(j, i)] = 1.0;
        }
        let lo = detect_modules(&a, ModulesAlgo::Louvain);
        let gr = detect_modules(&a, ModulesAlgo::Greedy);
        assert_eq!(lo.n_modules, 2, "sizes={:?}", lo.members.iter().map(Vec::len).collect::<Vec<_>>());
        assert_eq!(gr.n_modules, 2);
        assert_eq!(lo.module_of.len(), 6);
        assert_eq!(lo.roles.len(), 6);
        assert!((lo.modularity - 0.5).abs() < 1e-12, "Q={}", lo.modularity);
        // the enum parses for the CLI and defaults to Louvain
        assert_eq!(ModulesAlgo::from_str("LOUVAIN").unwrap(), ModulesAlgo::Louvain);
        assert_eq!(ModulesAlgo::from_str("greedy").unwrap(), ModulesAlgo::Greedy);
        assert!(ModulesAlgo::from_str("spectral").is_err());
        assert_eq!(ModulesAlgo::default(), ModulesAlgo::Louvain);
    }

    #[test]
    fn bootstrap_stability_recovers_planted_modules() {
        // two strong planted co-abundance modules: resampling 20 samples must
        // keep them essentially intact
        let d = crate::simulate::simulate(120, 20, 2, 4, 0, 0.2, 7);
        let samples: Vec<usize> = (0..d.abundance.ncols()).collect();
        let rep = bootstrap_stability(&d.abundance, &samples, 0.5, ModulesAlgo::Louvain, 20, 11);
        assert_eq!(rep.n_boot, 20);
        assert_eq!(rep.per_module_jaccard.len(), 2, "{:?}", rep.per_module_jaccard);
        assert!(
            rep.mean_jaccard > 0.8,
            "planted modules should be stable, mean Jaccard = {:.3} (per module {:?})",
            rep.mean_jaccard,
            rep.per_module_jaccard
        );
    }

    #[test]
    fn bootstrap_stability_collapses_on_noise() {
        // noise = 1.0 drowns the planted signal: the recovered partition is a
        // property of this sample draw and must not look stable
        let d = crate::simulate::simulate(120, 20, 2, 4, 4, 1.0, 7);
        let samples: Vec<usize> = (0..d.abundance.ncols()).collect();
        let rep = bootstrap_stability(&d.abundance, &samples, 0.5, ModulesAlgo::Louvain, 20, 11);
        assert_eq!(rep.n_boot, 20);
        assert!(
            rep.mean_jaccard < 0.5,
            "noise partition should not be stable, mean Jaccard = {:.3} (per module {:?})",
            rep.mean_jaccard,
            rep.per_module_jaccard
        );
    }
}
