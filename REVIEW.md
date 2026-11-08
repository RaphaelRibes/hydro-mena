# hydro-mena — review, fixes and additions (2026-11-08)

> **Correction (applied after the first pass).** `network::modularity` was
> mathematically wrong in both the original code and my first rewrite: it
> summed `1 - k_i k_j/2E` over **existing edges only**, omitting the
> configuration-model term for non-edges, which returns `1/n` instead of `0`
> for a complete graph in one community. It now uses the aggregate form
> `Q = Σ_c [L_c/(2E) − (d_c/2E)²]`, with known-answer tests (complete graph → 0,
> two disjoint triangles → 0.5). Every modularity figure in this document has
> been recomputed; the §2 comparison table and the `--st 0.52` values below
> supersede the first-pass numbers. The corrected figures are **lower** than
> first reported (ox 0.650, not 0.705), which strengthens rather than weakens
> the "the RMT stage was broken" conclusion.

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

| | before (rank unfolding) | after (corrected GOF, current build) |
|---|---|---|
| st | 0.010 | 0.890 |
| nodes / edges | 200 / 19 500 (complete) | 75 / 105 |
| avg degree | 195.0 | 2.8 |
| modularity M | 0.009 | 0.683 |
| γ (power law) | **−42.4** | 1.09 |
| roles | all "connector" | mixed |

The corrected scan is **not yet stable** across small changes to degeneracy
handling: an earlier build with a different tie tolerance landed at st=0.93 on
31 nodes, and one variant kept 21 nodes with 15 edges. A threshold chosen on a
20-node remnant is not a scientific result (see §6.1). What *is* established is
the negative claim: the old statistic was identically zero and accepted any
graph, including the complete one.

A negative γ and 100 % connectors are the tell: every downstream stage was
reading a saturated graph. This is why the 2025 repo's author hardcoded 0.52 and
why `--st` had to be added; the root cause was never the data.

Fixed by implementing what the paper specifies: fit the integrated density
N(λ) with a degree-5 polynomial, take spacings of N at consecutive eigenvalues,
normalise to unit mean, then a **binned** χ² GOF against Poisson (bins merged
until expected ≥ 5, dof = bins − 2). `ScanPoint` now carries `p`, `edges`,
`chi2`, `dof`, `crit`, `ks`.

## 2. Threshold behaviour depends on the degeneracy policy

On the real `ox` data the outcome **depends on the degeneracy policy** — this is
the most fragile part of the build and must be read as experimental:

| build | ox result |
|---|---|
| before eigenvalue dedup | no acceptance → fallback st=0.50 (168/173, E=910) |
| current (ties collapsed, ≥20 spacings) | **converges st=0.70** (140/173, E=361, M=0.845) |

Deng's MENA does not drop degenerate eigenvalues; collapsing ties at 1e-9 is my
choice, and it moves the accepted threshold by 0.2. The scan is also weakly
powered: at n=960 it produces **0** valid points and falls back on a 19-node
remnant, and at n=1920 it "converges" on a **single** scan point. A 20-node
remnant passing a goodness-of-fit test means little, so no threshold-dependent
claim should be published from this build. `rmt_scan.csv` now records `dof` and
per-point χ² so low-power points are visible.

Against the 2025 repo's reference network (`adjacency_matrix_ox.csv`,
`data/network_indexes_ox.json`):

| | repo | ours @ st=0.52 (fixed) | ours (RMT scan) |
|---|---|---|---|
| threshold | 0.52 | 0.52 | 0.700 (converged) |
| nodes / edges | 173 / 910 | 166 / 836 | 140 / 361 |
| modularity | 0.685 | 0.650 | 0.845 |
| avg clustering | 0.618 | 0.640 | — |
| GD | 3.80 | 3.77 | — |

With modularity corrected, ours is now **systematically below** the reference on
the fixed-threshold path (0.650 vs 0.685), where the first pass reported 0.705
and a near-match. The reference uses networkx, whose modularity is the correct
aggregate form, so this comparison is now apples-to-apples and the gap is real
(different community algorithm: greedy Louvain phase-1 here vs their detector).

The unpinned RMT scan converges at st=0.70 and returns a much sparser graph
(361 edges), so it is **not** comparable to the repo's fixed-0.52 network.

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
- `cargo test --release`: **40 tests pass** (was: no tests at all). Covers the
  eigensolver against a known spectrum, similarity ≡ |correlation|, clustering on
  a triangle and a dumbbell, modularity recovering two groups, BFS path length,
  Poisson GOF accepting exponential and rejecting flat spacings, polynomial
  unfolding, Mantel detecting a planted correlation, Clauset power-law MLE
  recovering γ on a synthetic power law and rejecting an exponential tail,
  and bootstrap module-stability plumbing.
- End-to-end on synthetic and on `real/abundance_{ox}.csv`.
- `viz.py all` produces 7 PNG+SVG; GUI serves `/api/runs`, `/api/run` and the page
  (verified with curl).

**Measured on the desktop** (`cachyos-x8664`, Ryzen 7 5800X3D, 16 threads,
RTX 2080 Ti via Vulkan, commit `1e8e439`, min-of-3, artifacts off): the CPU
pipeline runs 0.0017 s at n=60 and **55.1 s at n=1920**, of which **RMT is
55.0 s**. The corrected threshold makes the graphs sparse, so topology is now
~0 s and the eigendecompositions are the whole cost — the bottleneck moved.
Similarity: CPU 1.499 s vs GPU 1.523 s at n=12000, i.e. **0.98× — parity, not a
win**, with maxdiff 2.3e-7…7.9e-7 (PASS). The old "4.71× at n=12000" is
obsolete because the *CPU* path was parallelised (16 threads; the ~4.7× vs its
serial baseline is **inferred** from the old 6.98 s timing and was not re-timed
here), not because the GPU regressed; the GPU still pays a
fixed ~0.35 s. Full matrix with provenance in
`hydro-mena-vault/05-Benchmarks/bench-2026-11-08.md`.

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

## 7. Post-review fixes (uncommitted, 2026-11-08)

Working-tree only — no commit. Items 7.1–7.5 and 7.7–7.8 were implemented by
parallel workers on the same checkout and have **not** been re-run here: they are
documented as *pending merge + verification*. Item 7.6 (GPU kernel) was written
and measured on the desktop RTX 2080 Ti in this session.

### 7.1 Louvain is the default module algorithm (`--modules louvain|greedy`)

Greedy phase-1 agglomeration stops at the first local optimum and over-splits
sparse networks — the module counts elsewhere (9 modules on the seed-3 demo,
51 on the 480-OTU bench draw) are largely an artefact of that. Louvain's
local-moving + aggregation passes are now the default; `--modules greedy` keeps
the old path for comparison. Convergence is still single-start, so repeated runs
should be used to check partition stability (roadmap item 4). *Pending merge +
verify.*

### 7.2 CLR transform (`--clr`, `--clr-pseudocount`, default 1.0)

Pearson correlation on raw relative abundances is biased by the closure
constraint (rows sum to a constant), producing spurious negative correlations.
`--clr` applies the centred log-ratio `log(x + pc) − mean(log(x + pc))` per
sample before standardization/correlation; `pc` keeps zeros finite. Off by
default so count-based workflows are unchanged. *Pending merge + verify.*

### 7.3 Null z-scores (`--null-samples N`, default 0)

Modularity (and avgCC/GD) are now compared against `N` Maslov–Sneppen
degree-preserving rewired networks; `report.json` gains a `null` object with
`modularity_z`, `avg_cc_z`, `gd_z` (plus the sample count). This is what turns
"M = 0.65" into "M = 0.65, z = … against a degree-matched null". Default 0 = off.
*Pending merge + verify.*

### 7.4 RMT diagnostic columns: `brody_beta`, `ks_goe`

`rmt_scan.csv` now carries two spectral diagnostics per candidate threshold:
the Brody parameter β (Poisson NNSD β=0 → GOE/Wigner β=1) and the KS distance to
the GOE surmise (`ks_goe`). They make the Poisson-vs-chaotic decision auditable
instead of relying on the χ² p-value alone. *Pending merge + verify.*

### 7.5 Lanczos bulk spectrum for RMT at n > 512

The scan dense-eigendecomposes the kept subgraph at every candidate threshold;
that is the 55 s at n=1920 in §5. For n > 512 the bulk spectrum is now estimated
with Lanczos (a few hundred mat-vecs) instead of a full O(n³) decomposition. The
`n ≤ 512` path is unchanged so the small-n results above remain reproducible.
*Pending merge + verify.*

### 7.6 GPU similarity kernel: 16×16 tiles, vec4, cached device

`src/gpu.rs` rewritten: 16×16 output tiles per 16×16 workgroup with
`vec4<f32>` shared-memory staging (64 floats per k-slice) and host-side row
scaling that folds the 1/(m−1) denominator into the shader's dot product. The
8×8-per-thread kernel did one full M-loop per output element with scalar loads.
Row-chunks still keep every buffer under the 128 MiB binding limit, but all
chunks now share one pipeline and one command buffer (single submit/poll), and
the adapter/device/pipeline are created once per process (the ~0.25 s Vulkan
setup used to be paid on every call, dominating every n below ~10000).

| n×m | before, published (bench.csv) | before, committed kernel re-run | after, warm |
|---|---|---|---|
| 4000×64 | — | 0.372 s | **0.06 s** |
| 8000×96 | 0.647 s | 0.566 s | **0.24 s** |
| 12000×128 | 1.523 s | 1.038 s | **0.53 s** |

Warm = second and later calls in one process; a one-shot call still pays
~0.15–0.25 s device setup (≈0.75 s at n=12000, vs 1.5 s before). Parity vs the
CPU f64 path: max |diff| ≤ 3.84e-7 over parity_sizes (n=2048…3000, m=8),
parity_3000 (n=480…3000, m=64) and the m=96/128 sizes — one order of magnitude
inside the 1e-5 gate; m=5/6/7/13 and constant-row edge cases give ≤1.4e-7 and no
NaN. `cargo build --release` clean, `cargo test --release` 19/19 pass (no logic
changed outside the GPU path).

### 7.7 Ragged-CSV hard error

`io` now rejects abundance/trait files whose rows disagree with the header
(missing or extra fields) and trait tables that omit samples present in the
abundance table. Previously the reader truncated to the shorter row/column and
`pearson()` silently paired the wrong samples (§3 item 3). *Pending merge +
verify.*

### 7.8 Examples crate-rename fix

The GPU examples import the crate by its current name (`mena::…`) after the
package rename, and each GPU example declares `required-features = ["gpu"]` in
`Cargo.toml`, so a default `cargo test` checkout no longer tries to compile them.
*Pending merge + verify.*
