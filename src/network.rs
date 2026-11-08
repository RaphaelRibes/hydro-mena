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
use crate::modules::ModulesAlgo;
use crate::stats;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;
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
    /// Clauset MLE power-law fit of the degree tail: (gamma, xmin, ks, p)
    pub clauset_gamma: f64,
    pub clauset_xmin: f64,
    pub clauset_ks: f64,
    pub clauset_p: f64,
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

// ---------------------------------------------------------------------------
// Louvain (Blondel et al. 2008) — full two-phase, multilevel optimisation.
//
// `greedy_modularity` above is only phase 1 of one level: it cannot undo a bad
// early merge, so dense networks fragment into many small modules. Louvain
// alternates local moving with aggregation and repeats on the coarsened graph,
// which is both more accurate and the de-facto reference algorithm.
// ---------------------------------------------------------------------------

/// One level of the Louvain hierarchy: a weighted graph whose nodes are
/// supernodes (single original nodes at level 0). Edges *inside* a supernode
/// are not stored on purpose — they stay internal whichever community the
/// supernode joins, so the same constant appears in every candidate's ΔQ and
/// cancels out of the argmax. `k[i]` is the summed degree of the original
/// nodes inside supernode i, and `m2` (= Σ k = 2m) is invariant across levels.
struct LevelGraph {
    adj: Vec<Vec<(usize, f64)>>,
    k: Vec<f64>,
    m2: f64,
}

impl LevelGraph {
    fn from_adjacency(adj: &[Vec<usize>]) -> Self {
        let n = adj.len();
        let mut g = LevelGraph {
            adj: vec![Vec::new(); n],
            k: vec![0.0; n],
            m2: 0.0,
        };
        for i in 0..n {
            g.k[i] = adj[i].len() as f64;
            g.m2 += g.k[i];
            g.adj[i] = adj[i].iter().map(|&j| (j, 1.0)).collect();
        }
        g
    }
}

/// Relabel communities 0..K-1 in order of first appearance; returns K.
fn compact_labels(assign: &mut [usize]) -> usize {
    let mut map: std::collections::HashMap<usize, usize> = std::collections::HashMap::new();
    let mut next = 0usize;
    for a in assign.iter_mut() {
        *a = *map.entry(*a).or_insert_with(|| {
            let v = next;
            next += 1;
            v
        });
    }
    next
}

/// Phase 1 on a single level: sweep the nodes in random order, moving each to
/// the neighbouring community with the largest modularity gain
///
/// ΔQ ∝ k_i,in(c) − k_i · Σtot(c) / 2m,
///
/// with the node temporarily removed from its community so `Σtot(c)` excludes
/// it. The 2/m factor is common to every candidate and dropped.
/// Returns true if any node moved.
fn local_moving(g: &LevelGraph, assign: &mut [usize], rng: &mut StdRng) -> bool {
    let n = g.adj.len();
    let mut tot = vec![0.0f64; n];
    for i in 0..n {
        tot[assign[i]] += g.k[i];
    }
    // Scratch: weight from the current node into each community, with a sparse
    // reset list so a sweep stays O(E) instead of O(n·K).
    let mut into = vec![0.0f64; n];
    let mut touched: Vec<usize> = Vec::new();
    let mut order: Vec<usize> = (0..n).collect();
    let mut any = false;
    for _sweep in 0..1000 {
        order.shuffle(rng);
        let mut moved = false;
        for &i in &order {
            let ki = g.k[i];
            if ki == 0.0 {
                continue;
            }
            let cur = assign[i];
            touched.clear();
            for &(j, w) in &g.adj[i] {
                let c = assign[j];
                if into[c] == 0.0 {
                    touched.push(c);
                }
                into[c] += w;
            }
            tot[cur] -= ki;
            // Gain of staying (rejoining `cur` with i excluded) vs. each other
            // neighbouring community. Strict `>` keeps ties deterministic.
            let mut best_c = cur;
            let mut best_gain = into[cur] - ki * tot[cur] / g.m2;
            for &c in &touched {
                if c == cur {
                    continue;
                }
                let gain = into[c] - ki * tot[c] / g.m2;
                if gain > best_gain + 1e-12 {
                    best_gain = gain;
                    best_c = c;
                }
            }
            tot[best_c] += ki;
            if best_c != cur {
                assign[i] = best_c;
                moved = true;
                any = true;
            }
            for &c in &touched {
                into[c] = 0.0;
            }
        }
        if !moved {
            break;
        }
    }
    any
}

/// Phase 2: build the coarsened graph whose node `c` is community `c` of
/// `assign`. Edge weights are summed; intra-community edges are dropped (they
/// are internal to the supernode and never influence another move).
fn aggregate(g: &LevelGraph, assign: &[usize], k_com: usize) -> LevelGraph {
    use std::collections::HashMap;
    let mut maps: Vec<HashMap<usize, f64>> = vec![HashMap::new(); k_com];
    let mut k = vec![0.0f64; k_com];
    for i in 0..g.adj.len() {
        let ci = assign[i];
        k[ci] += g.k[i];
        for &(j, w) in &g.adj[i] {
            let cj = assign[j];
            if cj != ci {
                *maps[ci].entry(cj).or_insert(0.0) += w;
            }
        }
    }
    let adj: Vec<Vec<(usize, f64)>> = maps
        .into_iter()
        .map(|m| {
            let mut v: Vec<(usize, f64)> = m.into_iter().collect();
            // HashMap iteration order is randomized per process; sorting keeps
            // tie-breaking (and therefore the whole result) seed-deterministic.
            v.sort_unstable_by_key(|e| e.0);
            v
        })
        .collect();
    LevelGraph { adj, k, m2: g.m2 }
}

/// Full Louvain multilevel modularity optimisation on an adjacency list.
///
/// Phase 1 (local moving) + phase 2 (aggregation) are repeated until a level
/// fails to merge anything. Deterministic for a given `seed`: node order comes
/// from a seeded `StdRng` and every aggregated adjacency list is sorted, so no
/// HashMap iteration order can leak into the partition.
/// Returns (Q on the original graph, community id per original node).
pub fn louvain_adj(adj: &[Vec<usize>], seed: u64) -> (f64, Vec<usize>) {
    let n = adj.len();
    if n == 0 {
        return (0.0, Vec::new());
    }
    let m2: f64 = adj.iter().map(|a| a.len()).sum::<usize>() as f64;
    if m2 == 0.0 {
        return (0.0, vec![0; n]);
    }
    let mut rng = StdRng::seed_from_u64(seed);
    let mut g = LevelGraph::from_adjacency(adj);
    // mapping[o] = community of original node o at the current level.
    let mut mapping: Vec<usize> = (0..n).collect();
    for _level in 0..64 {
        let mut assign: Vec<usize> = (0..g.adj.len()).collect();
        local_moving(&g, &mut assign, &mut rng);
        let k_com = compact_labels(&mut assign);
        for o in mapping.iter_mut() {
            *o = assign[*o];
        }
        // No merge happened (each supernode is its own community) or all is
        // one community: further aggregation cannot improve Q.
        if k_com <= 1 || k_com == g.adj.len() {
            break;
        }
        g = aggregate(&g, &assign, k_com);
    }
    compact_labels(&mut mapping);
    let q = modularity(adj, &mapping, m2);
    (q, mapping)
}

/// Louvain multilevel modularity on a 0/1 adjacency matrix (the pipeline's
/// form). See [`louvain_adj`]; `seed` makes the result reproducible.
pub fn louvain(adj: &Mat, seed: u64) -> (f64, Vec<usize>) {
    louvain_adj(&adjacency_list(adj), seed)
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
    // BTreeMap (not HashMap): iteration order feeds the floating-point sum, and
    // per-process HashMap randomization made Q differ in the last bit between
    // identical runs. Sorted keys keep reported Q bit-reproducible.
    let mut tot = std::collections::BTreeMap::<usize, f64>::new();
    let mut internal = std::collections::BTreeMap::<usize, f64>::new();
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
/// not run the optimiser a second time for module roles). Defaults to Louvain;
/// use [`topology_parts_with`] to select the optimiser (e.g. for an ablation).
pub fn topology_parts(a: &Mat) -> (Topology, Vec<usize>) {
    topology_parts_with(a, ModulesAlgo::default())
}

/// [`topology_parts`] with an explicit modularity optimiser, so the reported
/// `modularity`/`n_modules` match the partition `detect_modules` returns.
pub fn topology_parts_with(a: &Mat, algo: ModulesAlgo) -> (Topology, Vec<usize>) {
    let adj = adjacency_list(a);
    let n = adj.len();
    let edges = adj.iter().map(|x| x.len()).sum::<usize>() / 2;
    let k: Vec<f64> = adj.iter().map(|x| x.len() as f64).collect();
    let avg_k = stats::mean(&k);
    let (gd, connected_frac) = avg_path_length(&adj);
    let cc = clustering_coefficients(&adj);
    let avg_cc = stats::mean(&cc);
    let (m, assign) = match algo {
        ModulesAlgo::Louvain => louvain_adj(&adj, 42),
        ModulesAlgo::Greedy => greedy_modularity(&adj),
    };
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

    // Clauset MLE + KS p-value on the degree sequence (defensible scale-free test)
    let deg_seq: Vec<usize> = k.iter().map(|&x| x as usize).collect();
    let (clauset_gamma, clauset_xmin, clauset_ks, clauset_p) =
        crate::stats::clauset_powerlaw(&deg_seq, 42, 100);

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
        clauset_gamma,
        clauset_xmin,
        clauset_ks,
        clauset_p,
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

    // --- Louvain known-answer tests -------------------------------------

    fn mat_from_edges(n: usize, edges: &[(usize, usize)]) -> Mat {
        let mut a = Mat::zeros((n, n));
        for &(i, j) in edges {
            a[(i, j)] = 1.0;
            a[(j, i)] = 1.0;
        }
        a
    }

    fn n_communities(assign: &[usize]) -> usize {
        let mut v = assign.to_vec();
        v.sort_unstable();
        v.dedup();
        v.len()
    }

    #[test]
    fn louvain_splits_two_disjoint_cliques() {
        let mut edges = Vec::new();
        for base in [0usize, 5] {
            for i in base..base + 5 {
                for j in (i + 1)..base + 5 {
                    edges.push((i, j));
                }
            }
        }
        let a = mat_from_edges(10, &edges);
        let (q, assign) = louvain(&a, 42);
        assert_eq!(n_communities(&assign), 2, "assign={assign:?}");
        assert!(assign[..5].windows(2).all(|w| w[0] == w[1]));
        assert!(assign[5..].windows(2).all(|w| w[0] == w[1]));
        assert_ne!(assign[0], assign[5]);
        assert!(q > 0.35, "Q={q}");
        // two disjoint K5 have exactly Q = 0.5
        assert!((q - 0.5).abs() < 1e-12, "two K5: Q must be 0.5, got {q}");
        let adj = adjacency_list(&a);
        let m2 = adj.iter().map(|x| x.len()).sum::<usize>() as f64;
        assert!((modularity(&adj, &assign, m2) - q).abs() < 1e-12);
    }

    /// Zachary's karate club: 34 nodes, 78 edges (canonical 0-indexed list).
    const KARATE_EDGES: [(usize, usize); 78] = [
        (0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (0, 7), (0, 8),
        (0, 10), (0, 11), (0, 12), (0, 13), (0, 17), (0, 19), (0, 21), (0, 31),
        (1, 2), (1, 3), (1, 7), (1, 13), (1, 17), (1, 19), (1, 21), (1, 30),
        (2, 3), (2, 7), (2, 8), (2, 9), (2, 13), (2, 27), (2, 28), (2, 32),
        (3, 7), (3, 12), (3, 13),
        (4, 6), (4, 10),
        (5, 6), (5, 10), (5, 16),
        (6, 16),
        (8, 30), (8, 32), (8, 33),
        (9, 33),
        (13, 33),
        (14, 32), (14, 33),
        (15, 32), (15, 33),
        (18, 32), (18, 33),
        (19, 33),
        (20, 32), (20, 33),
        (22, 32), (22, 33),
        (23, 25), (23, 27), (23, 29), (23, 32), (23, 33),
        (24, 25), (24, 27), (24, 31),
        (25, 31),
        (26, 29), (26, 33),
        (27, 33),
        (28, 31), (28, 33),
        (29, 32), (29, 33),
        (30, 32), (30, 33),
        (31, 32), (31, 33),
        (32, 33),
    ];

    #[test]
    fn louvain_karate_club_known_answer() {
        let a = mat_from_edges(34, &KARATE_EDGES);
        let (q, assign) = louvain(&a, 42);
        let k = n_communities(&assign);
        assert!((2..=4).contains(&k), "expected 2-4 communities, got {k}");
        // Reference Louvain partitions of karate land at Q ≈ 0.42.
        assert!((0.35..=0.45).contains(&q), "karate Q out of range: {q}");
        // seed-determinism and agreement with the shared Q helper
        let (q2, assign2) = louvain(&a, 42);
        assert_eq!(assign, assign2);
        assert!((q - q2).abs() < 1e-12);
        let adj = adjacency_list(&a);
        let m2 = adj.iter().map(|x| x.len()).sum::<usize>() as f64;
        assert!((modularity(&adj, &assign, m2) - q).abs() < 1e-12);
    }

    #[test]
    fn louvain_handles_degenerate_graphs() {
        // empty graph
        let (q, assign) = louvain(&Mat::zeros((0, 0)), 7);
        assert_eq!(q, 0.0);
        assert!(assign.is_empty());

        // single node
        let (q, assign) = louvain(&Mat::zeros((1, 1)), 7);
        assert_eq!(q, 0.0);
        assert_eq!(assign, vec![0]);

        // several nodes, no edges
        let (q, assign) = louvain(&Mat::zeros((4, 4)), 7);
        assert_eq!(q, 0.0);
        assert_eq!(assign.len(), 4);
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
