//! Network topology indices (paper Table 1): connectivity, average path length
//! (GD), average clustering coefficient, modularity, scale-free R^2 (power law
//! of the degree distribution), hierarchy R^2 (C(k) ~ k^-gamma).
//!
//! Two hot loops were asymptotically wrong and are fixed here:
//!   * clustering tested edge existence with `Vec::contains` inside a double
//!     loop over every neighbour pair — O(k^3) per node. Now a sorted-list
//!     intersection, O(k^2).
//!   * the modularity optimiser recomputed every community's total degree by
//!     scanning all n nodes for *each candidate move* — O(n^2) per pass. Totals
//!     are now maintained incrementally (Louvain phase-1 style).
//! Breadth-first search runs in parallel over source nodes with per-thread
//! scratch buffers.
use crate::linalg::Mat;
use crate::stats;
use rayon::prelude::*;
use std::collections::BTreeMap;

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
    /// edges / possible edges
    pub density: f64,
    pub n_components: usize,
    /// fraction of nodes in the largest connected component
    pub giant_frac: f64,
    /// fraction of ordered pairs that are connected (GD is computed over these)
    pub connected_frac: f64,
}

/// Build the adjacency list from a symmetric 0/1 matrix.
pub fn adjacency_list(a: &Mat) -> Vec<Vec<usize>> {
    let n = a.nrows();
    let mut adj = vec![Vec::new(); n];
    for i in 0..n {
        for j in (i + 1)..n {
            if a[(i, j)] > 0.0 {
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

/// Average shortest path length over reachable pairs (GD), in parallel over
/// source nodes. Each rayon worker reuses one distance and one queue buffer.
/// Returns (gd, connected_fraction).
pub fn avg_path_length(adj: &[Vec<usize>]) -> (f64, f64) {
    let n = adj.len();
    if n < 2 {
        return (f64::NAN, 0.0);
    }
    let (total, count) = (0..n)
        .into_par_iter()
        .fold(
            || (0u64, 0u64, vec![u32::MAX; n], Vec::<u32>::with_capacity(n)),
            |(mut tot, mut cnt, mut dist, mut queue), s| {
                for d in dist.iter_mut() {
                    *d = u32::MAX;
                }
                queue.clear();
                dist[s] = 0;
                queue.push(s as u32);
                let mut head = 0usize;
                while head < queue.len() {
                    let u = queue[head] as usize;
                    head += 1;
                    let du = dist[u];
                    for &v in &adj[u] {
                        if dist[v] == u32::MAX {
                            dist[v] = du + 1;
                            queue.push(v as u32);
                        }
                    }
                }
                for t in (s + 1)..n {
                    if dist[t] != u32::MAX {
                        tot += dist[t] as u64;
                        cnt += 1;
                    }
                }
                (tot, cnt, dist, queue)
            },
        )
        .map(|(t, c, _, _)| (t, c))
        .reduce(|| (0, 0), |a, b| (a.0 + b.0, a.1 + b.1));
    let gd = if count > 0 { total as f64 / count as f64 } else { f64::NAN };
    let possible = (n * (n - 1) / 2) as f64;
    (gd, if possible > 0.0 { count as f64 / possible } else { 0.0 })
}

/// Local clustering coefficients: for each node, the neighbours are sorted once
/// and the number of triangles is counted by list intersection, O(k^2) per node
/// instead of the previous O(k^3) `contains` scan.
pub fn clustering_coefficients(adj: &[Vec<usize>]) -> Vec<f64> {
    let mut sorted: Vec<Vec<usize>> = adj.to_vec();
    for l in sorted.iter_mut() {
        l.sort_unstable();
    }
    let n = sorted.len();
    let mut cc = vec![0.0f64; n];
    for i in 0..n {
        let k = sorted[i].len();
        if k < 2 {
            continue;
        }
        let ni = &sorted[i];
        let mut common = 0usize;
        for &u in ni {
            let nu = &sorted[u];
            // |N(i) ∩ N(u)|
            let (mut a, mut b) = (0usize, 0usize);
            while a < ni.len() && b < nu.len() {
                match ni[a].cmp(&nu[b]) {
                    std::cmp::Ordering::Less => a += 1,
                    std::cmp::Ordering::Greater => b += 1,
                    std::cmp::Ordering::Equal => {
                        common += 1;
                        a += 1;
                        b += 1;
                    }
                }
            }
        }
        // sum_u |N(i) ∩ N(u)| counts every triangle twice, so 2T_i = common,
        // and C_i = 2T_i / (k(k-1)) = common / (k(k-1)).
        cc[i] = common as f64 / (k * (k - 1)) as f64;
    }
    cc
}

/// Modularity optimisation, Louvain phase 1 with an incremental community-total
/// table: O(E) per sweep instead of O(n^2) per candidate move.
/// Returns (Q, community id per node) with compact labels.
pub fn greedy_modularity(adj: &[Vec<usize>]) -> (f64, Vec<usize>) {
    let n = adj.len();
    let m2: f64 = adj.iter().map(|a| a.len()).sum::<usize>() as f64; // 2L
    if m2 == 0.0 || n == 0 {
        return (0.0, vec![0; n]);
    }
    let k: Vec<f64> = adj.iter().map(|a| a.len() as f64).collect();
    let mut assign: Vec<usize> = (0..n).collect();
    // tot[c] = sum of degrees in community c (includes every node currently in c)
    let mut tot: Vec<f64> = k.clone();
    // scratch: degree from i into each community, with a sparse reset list
    let mut into = vec![0.0f64; n];
    let mut touched: Vec<usize> = Vec::new();
    let mut improved = true;
    let mut sweep = 0;
    while improved && sweep < 100 {
        improved = false;
        sweep += 1;
        for i in 0..n {
            let cur = assign[i];
            let ki = k[i];
            if ki == 0.0 {
                continue;
            }
            touched.clear();
            for &v in &adj[i] {
                let c = assign[v];
                if into[c] == 0.0 {
                    touched.push(c);
                }
                into[c] += 1.0;
            }
            // take i out of its community so totals exclude it
            tot[cur] -= ki;
            let mut best_c = cur;
            let mut best_gain = into[cur] - ki * tot[cur] / m2; // gain of staying (i as its own group is 0)
            for &c in &touched {
                let gain = into[c] - ki * tot[c] / m2;
                if gain > best_gain + 1e-12 {
                    best_gain = gain;
                    best_c = c;
                }
            }
            tot[best_c] += ki;
            if best_c != cur {
                assign[i] = best_c;
                improved = true;
            }
            for &c in &touched {
                into[c] = 0.0;
            }
        }
    }
    // compact labels
    let mut map: BTreeMap<usize, usize> = BTreeMap::new();
    let mut next = 0usize;
    for a in assign.iter_mut() {
        let e = map.entry(*a).or_insert_with(|| {
            let v = next;
            next += 1;
            v
        });
        *a = *e;
    }
    let q = modularity(adj, &assign, m2);
    (q, assign)
}

/// Modularity Q (Eq. 7).
///
/// Q = sum_c [ L_c / 2E - (d_c / 2E)^2 ], where L_c counts *directed* internal
/// edges of community c and d_c is its summed degree. The previous version
/// summed (1 - k_i k_j/2E) over existing edges only, which omits the
/// configuration-model term for **non-edges** (and the diagonal): it returns
/// 1/n instead of 0 for a complete graph in a single community, so every
/// reported modularity was inflated.
pub fn modularity(adj: &[Vec<usize>], assign: &[usize], m2: f64) -> f64 {
    if m2 <= 0.0 {
        return 0.0;
    }
    let n = adj.len();
    let mut tot = std::collections::HashMap::<usize, f64>::new();
    let mut internal = std::collections::HashMap::<usize, f64>::new();
    for i in 0..n {
        let k = adj[i].len() as f64;
        *tot.entry(assign[i]).or_insert(0.0) += k;
        for &j in &adj[i] {
            if assign[j] == assign[i] {
                *internal.entry(assign[i]).or_insert(0.0) += 1.0;
            }
        }
    }
    let mut q = 0.0;
    for (c, &d) in &tot {
        let l = internal.get(c).copied().unwrap_or(0.0);
        q += l / m2 - (d / m2).powi(2);
    }
    q
}

/// Topology of the graph, plus the community partition it found (so callers do
/// not run the optimiser a second time for module roles).
pub fn topology_parts(a: &Mat) -> (Topology, Vec<usize>) {
    let adj = adjacency_list(a);
    let n = adj.len();
    let edges = adj.iter().map(|x| x.len()).sum::<usize>() / 2;
    let k: Vec<f64> = adj.iter().map(|x| x.len() as f64).collect();
    let avg_k = stats::mean(&k);
    let (gd, connected_frac) = avg_path_length(&adj);
    let cc = clustering_coefficients(&adj);
    let avg_cc = stats::mean(&cc);
    let (m, assign) = greedy_modularity(&adj);
    let n_modules = assign.iter().cloned().collect::<std::collections::HashSet<_>>().len();

    let comp = connected_components(&adj);
    let n_components = comp.iter().max().map(|m| m + 1).unwrap_or(0);
    let mut sizes = vec![0usize; n_components];
    for &c in &comp {
        sizes[c] += 1;
    }
    let giant = sizes.iter().copied().max().unwrap_or(0) as f64 / n.max(1) as f64;

    // scale-free: linear fit of log10 P(k) on log10 k
    let mut counts: BTreeMap<usize, usize> = BTreeMap::new();
    for &kv in &k {
        *counts.entry(kv as usize).or_insert(0) += 1;
    }
    let ks: Vec<f64> = counts.keys().map(|&x| x as f64).collect();
    let (r2_powerlaw, gamma) = if ks.len() >= 3 {
        let lx: Vec<f64> = ks.iter().map(|x| x.log10()).collect();
        let ly: Vec<f64> = ks
            .iter()
            .map(|x| (counts[&(*x as usize)] as f64 / n as f64).log10())
            .collect();
        let (r, slope) = linreg(&lx, &ly);
        (r * r, -slope)
    } else {
        (f64::NAN, f64::NAN)
    };

    // hierarchy: log C(k) on log k where C(k) > 0
    let pts: Vec<(f64, f64)> = (0..n)
        .filter(|&i| k[i] > 1.0 && cc[i] > 0.0)
        .map(|i| (k[i].log10(), cc[i].log10()))
        .collect();
    let r2_hierarchy = if pts.len() >= 3 {
        let lx: Vec<f64> = pts.iter().map(|p| p.0).collect();
        let ly: Vec<f64> = pts.iter().map(|p| p.1).collect();
        let (r, _) = linreg(&lx, &ly);
        r * r
    } else {
        f64::NAN
    };

    let density = if n > 1 {
        2.0 * edges as f64 / (n * (n - 1)) as f64
    } else {
        0.0
    };

    let topo = Topology {
        n,
        edges,
        avg_k,
        gd,
        avg_cc,
        modularity: m,
        n_modules,
        r2_powerlaw,
        gamma,
        r2_hierarchy,
        density,
        n_components,
        giant_frac: giant,
        connected_frac,
    };
    (topo, assign)
}

pub fn topology(a: &Mat) -> Topology {
    topology_parts(a).0
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

#[cfg(test)]
mod tests {
    use super::*;

    /// two triangles joined by a single edge
    fn dumbell() -> Vec<Vec<usize>> {
        vec![
            vec![1, 2],
            vec![0, 2],
            vec![0, 1, 3],
            vec![2, 4, 5],
            vec![3, 5],
            vec![3, 4],
        ]
    }

    #[test]
    fn clustering_of_a_triangle_is_one() {
        let adj = vec![vec![1, 2], vec![0, 2], vec![0, 1]];
        let cc = clustering_coefficients(&adj);
        for v in cc {
            assert!((v - 1.0).abs() < 1e-12);
        }
    }

    #[test]
    fn clustering_of_dumbell() {
        let cc = clustering_coefficients(&dumbell());
        // nodes 2 and 3 sit in one triangle each and have a third, non-closed edge
        assert!((cc[2] - 1.0 / 3.0).abs() < 1e-12, "{cc:?}");
        assert!((cc[0] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn modularity_of_complete_graph_is_zero() {
        // A complete graph in a single community has Q = 0 by definition.
        // The old edge-only sum returned 1/n here, which is why every reported
        // modularity was inflated.
        let n = 6;
        let mut adj = vec![Vec::new(); n];
        for i in 0..n {
            for j in 0..n {
                if i != j {
                    adj[i].push(j);
                }
            }
        }
        let assign = vec![0usize; n];
        let m2: f64 = adj.iter().map(|a| a.len()).sum::<usize>() as f64;
        let q = modularity(&adj, &assign, m2);
        assert!(q.abs() < 1e-12, "complete graph Q must be 0, got {q}");
    }

    #[test]
    fn modularity_of_two_disjoint_triangles_is_half() {
        // two K3, one community each: Q = 2 * (6/12 - (6/12)^2) = 0.5
        let adj = vec![
            vec![1, 2],
            vec![0, 2],
            vec![0, 1],
            vec![4, 5],
            vec![3, 5],
            vec![3, 4],
        ];
        let assign = vec![0usize, 0, 0, 1, 1, 1];
        let q = modularity(&adj, &assign, 12.0);
        assert!((q - 0.5).abs() < 1e-12, "got {q}");
    }

    #[test]
    fn modularity_finds_the_two_groups() {
        let (q, assign) = greedy_modularity(&dumbell());
        assert!(q > 0.3, "Q={q}");
        assert_eq!(assign[0], assign[1]);
        assert_eq!(assign[0], assign[2]);
        assert_eq!(assign[3], assign[4]);
        assert_ne!(assign[0], assign[3]);
    }

    #[test]
    fn path_length_of_a_path_is_known() {
        // 0-1-2-3: distances 1,2,3,1,2,1 over 6 pairs
        let adj = vec![vec![1], vec![0, 2], vec![1, 3], vec![2]];
        let (gd, frac) = avg_path_length(&adj);
        assert!((gd - 10.0 / 6.0).abs() < 1e-12, "gd={gd}");
        assert!((frac - 1.0).abs() < 1e-12);
    }
}
