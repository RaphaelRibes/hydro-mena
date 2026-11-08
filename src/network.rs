//! Network topology indices (paper Table 1): connectivity, average path length
//! (GD), average clustering coefficient, modularity, scale-free R^2 (power law
//! of degree distribution), hierarchy R^2 (C(k) ~ k^-gamma).
use crate::linalg::Mat;
use crate::stats;

pub struct Topology {
    pub n: usize,
    pub edges: usize,
    pub avg_k: f64,
    pub gd: f64,
    pub avg_cc: f64,
    pub modularity: f64,
    pub n_modules: usize,
    pub r2_powerlaw: f64,
    pub gamma: f64,
    pub r2_hierarchy: f64,
}

/// Build adjacency list from symmetric 0/1 matrix.
pub fn adjacency_list(A: &Mat) -> Vec<Vec<usize>> {
    let n = A.nrows();
    let mut adj = vec![Vec::new(); n];
    for i in 0..n {
        for j in (i + 1)..n {
            if A[(i, j)] > 0.0 {
                adj[i].push(j);
                adj[j].push(i);
            }
        }
    }
    adj
}

fn connected_components(adj: &[Vec<usize>]) -> Vec<usize> {
    let n = adj.len();
    let mut comp = vec![usize::MAX; n];
    let mut c = 0;
    for s in 0..n {
        if comp[s] != usize::MAX {
            continue;
        }
        let mut stack = vec![s];
        comp[s] = c;
        while let Some(u) = stack.pop() {
            for &v in &adj[u] {
                if comp[v] == usize::MAX {
                    comp[v] = c;
                    stack.push(v);
                }
            }
        }
        c += 1;
    }
    comp
}

/// Average shortest path length over reachable pairs (GD).
pub fn avg_path_length(adj: &[Vec<usize>]) -> f64 {
    let n = adj.len();
    let mut total = 0.0;
    let mut count = 0u64;
    for s in 0..n {
        let mut dist = vec![usize::MAX; n];
        dist[s] = 0;
        let mut queue = std::collections::VecDeque::from(vec![s]);
        while let Some(u) = queue.pop_front() {
            for &v in &adj[u] {
                if dist[v] == usize::MAX {
                    dist[v] = dist[u] + 1;
                    queue.push_back(v);
                }
            }
        }
        for t in (s + 1)..n {
            if dist[t] != usize::MAX {
                total += dist[t] as f64;
                count += 1;
            }
        }
    }
    if count > 0 {
        total / count as f64
    } else {
        f64::NAN
    }
}

pub fn clustering_coefficients(adj: &[Vec<usize>]) -> Vec<f64> {
    let n = adj.len();
    let k: Vec<usize> = adj.iter().map(|a| a.len()).collect();
    (0..n)
        .map(|i| {
            if k[i] < 2 {
                0.0
            } else {
                let mut links = 0usize;
                for a in 0..adj[i].len() {
                    for b in (a + 1)..adj[i].len() {
                        let u = adj[i][a];
                        let v = adj[i][b];
                        if adj[u].contains(&v) {
                            links += 1;
                        }
                    }
                }
                2.0 * links as f64 / (k[i] * (k[i] - 1)) as f64
            }
        })
        .collect()
}

/// Greedy modularity optimization (Clauset-Newman-Moore), returns (M, assignment).
/// Assignment: module id per node. Uses the standard agglomerative approach.
pub fn greedy_modularity(adj: &[Vec<usize>]) -> (f64, Vec<usize>) {
    let n = adj.len();
    let m2 = adj.iter().map(|a| a.len()).sum::<usize>() as f64; // 2L
    if m2 == 0.0 || n == 0 {
        return (0.0, vec![0; n]);
    }
    let mut assign: Vec<usize> = (0..n).collect();
    let mut k: Vec<f64> = adj.iter().map(|a| a.len() as f64).collect();
    // e_ij fraction of edges between communities; maintain via delta-Q merge
    // Simpler: use Louvain-like one-level passes on the modularity matrix.
    // Here: iterative best-move (Louvain phase 1) which is equivalent in spirit
    // and deterministic.
    let mut improved = true;
    let mut iter = 0;
    while improved && iter < 100 {
        improved = false;
        iter += 1;
        for i in 0..n {
            let cur = assign[i];
            // degree of i into each neighboring community
            let mut into: std::collections::HashMap<usize, f64> = std::collections::HashMap::new();
            for &v in &adj[i] {
                *into.entry(assign[v]).or_insert(0.0) += 1.0;
            }
            let ki = k[i];
            let mut best_comm = cur;
            let mut best_gain = 0.0f64;
            let mut candidates: Vec<usize> = into.keys().cloned().collect();
            candidates.push(cur);
            candidates.sort();
            candidates.dedup();
            for &c in &candidates {
                if c == cur {
                    continue;
                }
                let kin = *into.get(&c).unwrap_or(&0.0);
                // remove i from cur, add to c: delta Q (up to factor 1/m2)
                let k_in_cur = *into.get(&cur).unwrap_or(&0.0) - 1.0; // links to cur minus self-loop none
                let _ = k_in_cur;
                let tot_c = community_degree(&assign, &k, c);
                let tot_cur = community_degree(&assign, &k, cur) - ki;
                let gain = (kin - ki * tot_c / m2) - (k_in_cur.max(0.0) - ki * tot_cur / m2);
                if gain > best_gain + 1e-12 {
                    best_gain = gain;
                    best_comm = c;
                }
            }
            if best_comm != cur {
                assign[i] = best_comm;
                improved = true;
            }
        }
    }
    // compact labels
    let mut map = std::collections::HashMap::new();
    let mut next = 0usize;
    for a in assign.iter_mut() {
        let e = map.entry(*a).or_insert_with(|| {
            let v = next;
            next += 1;
            v
        });
        *a = *e;
    }
    let M = modularity(adj, &assign, m2);
    (M, assign)
}

fn community_degree(assign: &[usize], k: &[f64], c: usize) -> f64 {
    assign
        .iter()
        .enumerate()
        .filter(|(_, &a)| a == c)
        .map(|(i, _)| k[i])
        .sum()
}

/// Modularity Q (Eq. 7).
pub fn modularity(adj: &[Vec<usize>], assign: &[usize], m2: f64) -> f64 {
    let n = adj.len();
    let mut q = 0.0;
    for i in 0..n {
        for &j in &adj[i] {
            if assign[i] == assign[j] {
                q += 1.0 - (adj[i].len() as f64 * adj[j].len() as f64) / m2;
            }
        }
    }
    q / m2
}

pub fn topology(A: &Mat) -> Topology {
    let adj = adjacency_list(A);
    let n = adj.len();
    let edges = adj.iter().map(|a| a.len()).sum::<usize>() / 2;
    let k: Vec<f64> = adj.iter().map(|a| a.len() as f64).collect();
    let avg_k = stats::mean(&k);
    let gd = avg_path_length(&adj);
    let cc = clustering_coefficients(&adj);
    let avg_cc = stats::mean(&cc);
    let (M, assign) = greedy_modularity(&adj);
    let n_modules = assign.iter().cloned().collect::<std::collections::HashSet<_>>().len();

    // scale-free: linear fit log10 P(k) vs log10 k
    let mut counts: std::collections::BTreeMap<usize, usize> = std::collections::BTreeMap::new();
    for &kv in &k {
        *counts.entry(kv as usize).or_insert(0) += 1;
    }
    let ks: Vec<f64> = counts.keys().map(|&x| x as f64).collect();
    let (r2_powerlaw, gamma) = if ks.len() >= 3 {
        let lx: Vec<f64> = ks.iter().map(|x| x.log10()).collect();
        let ly: Vec<f64> = ks
            .iter()
            .map(|x| (counts[&(*x as usize)] as f64 / n as f64).log10())
            .collect();        let (r, slope) = linreg(&lx, &ly);
        (r * r, -slope)
    } else {
        (f64::NAN, f64::NAN)
    };

    // hierarchy: log C(k) vs log k where C(k) > 0
    let mut pts: Vec<(f64, f64)> = Vec::new();
    for i in 0..n {
        if k[i] > 1.0 && cc[i] > 0.0 {
            pts.push((k[i].log10(), cc[i].log10()));
        }
    }
    let r2_hierarchy = if pts.len() >= 3 {
        let lx: Vec<f64> = pts.iter().map(|p| p.0).collect();
        let ly: Vec<f64> = pts.iter().map(|p| p.1).collect();
        let (r, _) = linreg(&lx, &ly);
        r * r
    } else {
        f64::NAN
    };

    let _ = connected_components(&adj);
    Topology {
        n,
        edges,
        avg_k,
        gd,
        avg_cc,
        modularity: M,
        n_modules,
        r2_powerlaw,
        gamma,
        r2_hierarchy,
    }
}

pub fn linreg(x: &[f64], y: &[f64]) -> (f64, f64) {
    let n = x.len() as f64;
    let mx = x.iter().sum::<f64>() / n;
    let my = y.iter().sum::<f64>() / n;
    let mut sxy = 0.0;
    let mut sxx = 0.0;
    let mut syy = 0.0;
    for i in 0..x.len() {
        sxy += (x[i] - mx) * (y[i] - my);
        sxx += (x[i] - mx).powi(2);
        syy += (y[i] - my).powi(2);
    }
    let slope = if sxx > 0.0 { sxy / sxx } else { 0.0 };
    let r = if sxx > 0.0 && syy > 0.0 { sxy / (sxx * syy).sqrt() } else { 0.0 };
    (r, slope)
}
