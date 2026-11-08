//! Module roles (Eq. 8-9): within-module connectivity zi, participation
//! coefficient Pi, and the 4-category classification (peripheral / connector /
//! module hub / network hub with zi<=2.5, Pi<=0.62 thresholds).
use crate::network::{adjacency_list, greedy_modularity, modularity};
use crate::linalg::Mat;
use std::collections::HashMap;

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

pub fn analyze_modules(a: &Mat) -> ModuleAnalysis {
    let adj = adjacency_list(a);
    let (m, assign) = greedy_modularity(&adj);
    analyze_modules_with(&adj, m, assign)
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
    order.sort_by_key(|c| std::cmp::Reverse(sizes[c]));
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
    let m2 = adj.iter().map(|a| a.len()).sum::<usize>() as f64;
    let (_, assign) = greedy_modularity(&adj);
    modularity(&adj, &assign, m2)
}
