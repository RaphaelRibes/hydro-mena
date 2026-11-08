# hydro-mena — review, fixes and additions (2026-11-08)

Scope: the `hydro-mena` Rust CLI (`~/hermes_scratch/hydro-mena`), the vault that
documents it (`hydro-mena-vault`), and the processes that produce its numbers.
Everything below is anchored to a file and a reproduced run.

## 1. Headline finding: the RMT stage was a no-op

`src/rmt.rs::unfolded_spacings` built "unfolded" spacings from the rank alone:

```rust
let e: Vec<f64> = (0..p).map(|i| (i as f64 + 0.5) / p as f64).collect();
d.push(p as f64 * (e[i + 1] - e[i]));   // == 1.0 for every i
```

Every spacing was exactly 1.0, so `stats::poisson_chi2` — itself just
`sum((d-1)^2)` — returned exactly 0 for **every** candidate threshold. The scan
therefore accepted the *first* threshold it tried. With `st0 = 0.01` (changed in
`code-changes-2026-11-08.md`) that is the complete graph.

Reproduced on 200 simulated OTUs, default `run`:

| | before | after |
|---|---|---|
| st | 0.010 | 0.930 |
| nodes / edges | 200 / 19 500 (complete) | 31 / 31 |
| avg degree | 195.0 | 2.00 |
| modularity M | 0.009 | 0.775 |
| γ (power law) | **−42.4** | 0.86 |
| roles | 200 × "connector" | 31 × "peripheral" |
| GD | 1.02 | 2.36 (8 components) |

A negative γ and 100 % connectors are the tell: every downstream stage was
reading a saturated graph. This is why the 2025 repo's author hardcoded 0.52 and
why `--st` had to be added; the root cause was never the data.

Fixed by implementing what the paper specifies: fit the integrated density
N(λ) with a degree-5 polynomial, take spacings of N at consecutive eigenvalues,
normalise to unit mean, then a **binned** χ² GOF against Poisson (bins merged
until expected ≥ 5, dof = bins − 2). `ScanPoint` now carries `p`, `edges`,
`chi2`, `dof`, `crit`, `ks`.

## 2. Honest behaviour when no threshold is accepted

On the real `ox` data no threshold passes at α = 0.01 — consistent with the vault
note `rmt-threshold.md`. The old code reported `converged = true` anyway
(`!log_coarse.is_empty()`) and returned the last scanned point. Now:

```
[rmt] st = 0.500 converged=false FALLBACK (14 scan points, kept 168/173, E=910)
```

and `report.json` carries `rmt.fallback: true`, which the GUI shows as a warning.
The fallback picks the point with the smallest χ²/critical ratio. Against the
2025 repo's reference network (`adjacency_matrix_ox.csv`,
`data/network_indexes_ox.json`):

| | repo | this run (fallback st=0.50) |
|---|---|---|
| modularity | 0.685 | 0.658 |
| avg clustering | 0.618 | 0.637 |
| GD | 3.80 | 3.80 |

The validated `--st 0.52` path still reproduces the earlier numbers
(n=166, E=836, M=0.705).

## 3. Other defects found

| # | Where | Problem | Status |
|---|---|---|---|
| 1 | `pipeline.rs` GPU block | the GPU similarity matrix was computed, written to `S.csv`, then **discarded**; the CPU matrix was recomputed and used, while reporting `gpu_used=true` | fixed: `similarity()` returns the GPU result |
| 2 | `pipeline.rs::bench_all` | benchmarked a *copy* of the pipeline (no Mantel, different RMT start), so numbers drifted from real runs | fixed: benchmarks call `run_pipeline` with `write_artifacts: false` |
| 3 | `io::read_traits_csv` | dropped samples missing from the trait file, returning fewer rows; `pearson()` truncates to the shorter vector, so every trait correlation silently paired the wrong samples | fixed: hard error on any missing/ragged sample |
| 4 | `network::clustering_coefficients` | `adj[u].contains(&v)` inside a double loop: O(k³) per node | O(k²) sorted-intersection |
| 5 | `network::greedy_modularity` | recomputed each community's total degree by scanning all n nodes per candidate move — O(n²) per pass | incremental totals, O(E) per sweep |
| 6 | `linalg::jacobi_eigen` | rescanned the whole matrix for the largest off-diagonal *per rotation* — O(n⁴) | Householder + implicit QR (nalgebra), O(n³) |
| 7 | `linalg::*` | serial O(n²m) loops | rayon over disjoint row blocks |
| 8 | `network::topology` | computed a partition, threw it away (`let _ = connected_components`), then `modules.rs` recomputed it | `topology_parts()` returns the partition; roles reuse it |
| 9 | `io` / CLI | `similarity.csv` (n² rows) always written | `--write-similarity`, off by default |
| 10 | `Cargo.toml` | `examples/*` import the gated `gpu` module with no `required-features`, so `cargo test` failed on a default checkout; `rand_chacha` unused | `required-features` added; dep dropped |
| 11 | `stats::mantel` | permuted the *condensed pair vector*, which destroys matrix structure and inflates significance | permutes node labels (standard null), reports null mean, rayon-parallel |
| 12 | `pipeline.rs` | `std::fs::write(csv_of(...))` duplicated the csv writer; `return_continue`/`false_marker` dead code | removed |

## 4. New features

- **Artifacts for tooling** (all written by `run`): `edges.csv`, `nodes.csv`
  (module, role, zi, Pi, degree, MM, strongest trait), `rmt_scan.csv`,
  `spectrum.csv`, `degree_dist.csv`, `modules.csv`, plus the existing
  `fig5_r/p.csv`, `eigengenes.csv`, `adjacency.csv`, `report.json`.
- `--rmt-start`, `--alpha`, `--write-similarity`.
- `report.json` records input paths, seed, permutations and whether the threshold
  was fixed — previously a run could not be traced back to its inputs.
- **`viz.py`** — seven figures from artifacts: Fig-5 heatmap with significance
  stars, force-layout network, zi–Pi role map, RMT scan diagnostic, spectrum +
  NNSD vs Poisson/Wigner, degree distribution with fit, module sizes/variance.
- **`gui/app.py`** — "MENA Studio", stdlib-only local web GUI: run browser,
  overview cards, RMT scan, heatmap, interactive network, roles, degrees,
  modules, and a form that launches a run through the binary.

## 5. Verification

- `cargo build --release` clean (8 pre-existing `non_snake_case` warnings).
- `cargo test --release`: **14 tests pass** (was: no tests at all). Covers the
  eigensolver against a known spectrum, similarity ≡ |correlation|, clustering on
  a triangle and a dumbbell, modularity recovering two groups, BFS path length,
  Poisson GOF accepting exponential and rejecting flat spacings, polynomial
  unfolding, Mantel detecting a planted correlation.
- End-to-end on synthetic and on `real/abundance_{ox}.csv`.
- `viz.py all` produces 7 PNG+SVG; GUI serves `/api/runs`, `/api/run` and the page
  (verified with curl).

**Not measured:** a like-for-like speedup table. The old timings in
`05-Benchmarks/cpu-scaling.md` were taken on the degenerate complete graph, so
they are not comparable; the fixes above are asymptotic changes, not tuning.
Run `hydro-mena bench` on the desktop for a fresh matrix before quoting numbers.

## 6. Roadmap

1. **Threshold policy.** Decide and document one: strict RMT, χ²-minimum, or the
   repo's fixed 0.52. The GUI's scan view exists to make that choice visible.
2. **Degree-distribution fit.** OLS on log-binned P(k) is what ecologists report,
   but a Clauset MLE + KS test is the defensible test of "scale-free".
3. **Negative/positive edges.** Only |r| is kept; signed co-occurrence networks
   distinguish competition from cooperation and are standard in the field.
4. **Bootstrap stability.** Resample samples, re-run, report module persistence
   (Jaccard) — currently a single point estimate with no uncertainty.
5. **`serve` subcommand** in the CLI to replace the Python GUI server.
6. **Bench matrix** with the corrected threshold, plus a GPU re-check at n ≥ 3000.
