# hydro-mena — RMT-based Molecular Ecological Network Analysis CLI

Reimplements the full MENA pipeline of **Deng et al., BMC Bioinformatics 2012, 13:113**
("Molecular ecological network analyses", the paper at `~/Downloads/1471-2105-13-113.pdf`),
whose Figure 5 is the module-eigengene × environmental-trait correlation heatmap.

**Architecture (per user directive): Rust for all compute, Python only for graphs, GPU support, pixi env.**

- `src/*.rs` — Rust library + CLI (cargo, release profile, LTO). CPU-parallel via rayon.
- `src/gpu.rs` — optional wgpu/Vulkan compute (NVIDIA RTX 2080 Ti on this desktop) for the O(n²m) similarity matrix. Build with `--features gpu`; CLI flag `--gpu`.
- `plot.py` — the ONLY Python: renders the Fig-5-style heatmap from the CSVs the Rust CLI writes (matplotlib/seaborn).
- `pixi.toml` — env (conda-forge: python 3.12 + matplotlib/seaborn/pandas/numpy, rust). Tasks: `pixi run simulate|run|bench`.

## Pipeline stages implemented (paper Methods)

1. **Standardization** — x_ik = (y_ik − mean_i)/σ_i (Methods).
2. **Pearson correlation** R, **similarity** S = |R| (Eq. 1–2). GPU-accelerated with `--gpu`. With `--clr`, rows are CLR-transformed first (see below).
3. **RMT threshold** (Fig. 2, steps a–k): scan st upward (0.1 coarse → 0.01 fine), build the **weighted** thresholded similarity matrix of the giant component (Eq. 3: A_ij = s_ij if s_ij ≥ st, else 0, zero diagonal), eigen-decompose (dense for n ≤ 512, Lanczos bulk spectrum for n > 512), unfold eigenvalues by local density average, χ² GOF of the NNSD against Poisson exp(−d) at α=0.01; first st where H0 is accepted is the threshold. `rmt_scan.csv` also reports the `brody_beta` and `ks_goe` diagnostics (Poisson β=0 ↔ GOE β=1, and the KS distance to the GOE surmise).
4. **Topology** (Table 1): n, edges, avgK, GD (avg shortest path over reachable pairs), avgCC, modularity M (Eq. 7), scale-free R² (log P(k) vs log k, γ), hierarchy R² (C(k) ~ k^−γ).
5. **Modules** — modularity optimization, **Louvain by default** (`--modules louvain|greedy`; greedy phase-1 is kept for comparison but over-splits); **roles** via within-module connectivity zi and participation coefficient Pi (Eq. 8–9) → peripheral / connector / module hub / network hub (zi≤2.5, Pi≤0.62).
6. **Eigengenes** (Eq. 10–12): SVD per module, E_b = first right singular vector, Φ_b = variance explained, MM_iE = cor(x_i, E_b).
7. **Figure 5** — Pearson r and p between each module eigengene and each standardized trait → heatmap (red positive / green negative, numbers = r with p in parentheses), rendered by `plot.py`.
8. **Trait links** — GS_i,h = cor(x_i,T_h)² (Eq. 13) + **Mantel test** (permutation) between connectivity distance |k_i−k_j| and GS distance.
9. **Random-network comparison**: Maslov–Sneppen double-edge-swap nulls in `network.rs`/`pipeline.rs`, exposed as `--null-samples N` (default 0); z-scores for modularity, average clustering and geodesic distance go into `report.json`'s `null` object.

## Compositionality and nulls

**CLR instead of raw Pearson (`--clr`, `--clr-pseudocount`, default 1.0).** Abundance tables are compositional: each sample sums to a fixed total, so the values are not free to vary independently and Pearson correlation between raw relative abundances is biased (spurious negative correlations are the classic symptom — the "closure constraint"). The centred log-ratio transform `clr(x)_ik = log(x_ik + pc) − mean_k log(x_i· + pc)` maps the row onto the simplex's tangent space, removing the constant-sum constraint before standardizing and correlating; `pc` keeps zeros finite. Use `--clr` when the input is relative abundance (proportions/percents) rather than already-normalized counts; the flag is off by default so existing CSV workflows are unchanged. The transformed values feed the same RMT/topology pipeline.

**Module algorithm (`--modules louvain|greedy`, default `louvain`).** The greedy phase-1 agglomeration used previously stops at the first local optimum and fragments sparse networks into many small modules (it reported 9–51 modules where Louvain finds a coarser partition). Louvain's local moves + aggregation pass is now the default; `--modules greedy` reproduces the old behaviour for comparison.

**Null z-scores (`--null-samples N`, default 0).** A modularity of 0.65 means nothing without a null: for a sparse graph it can arise from degree sequence alone. With `--null-samples N` the pipeline generates N Maslov–Sneppen degree-preserving rewired networks, recomputes topology on each, and writes z-scores `(observed − null_mean)/null_sd` for modularity, average clustering and geodesic distance. They are reported as a `null` object in `report.json`:

```json
"null": { "samples": 100, "modularity_z": 4.7, "avg_cc_z": 2.9, "gd_z": -0.4 }
```

Default 0 disables the nulls (no extra cost). A |z| below ~2 means the observed value is within the rewired ensemble's spread and should not be read as structure.

## Usage

```fish
cd ~/hermes_scratch/hydro-mena
export PATH="$HOME/.pixi/bin:$PATH"
pixi run simulate -- --otus 120 --samples 24 --modules 5 --traits 4 --noise 0.3 --seed 3 --out-dir demo_final
pixi run run -- run --abundance demo_final/abundance.csv --traits demo_final/traits.csv --out-dir demo_final/out \
    [--gpu] [--st 0.52] [--clr [--clr-pseudocount 1.0]] \
    [--modules louvain|greedy] [--null-samples N] [--signed] [--boot N]
pixi run run -- bench --out-dir benches            # CPU scaling; add --features gpu build for GPU similarity benches
pixi run python plot.py fig5 --r demo_final/out/fig5_r.csv --p demo_final/out/fig5_p.csv --out fig5.png --svg fig5.svg
```

GPU build: `cargo build --release --features gpu` (wgpu → Vulkan/RADV). `--gpu` on `run` uses the adapter for the similarity matrix and prints which adapter was used; falls back to CPU with a warning if unavailable. Verified after the 16×16-tile rework (2026-11-08): GPU vs CPU similarity max |diff| 1.6e-7…3.8e-7 at every tested size, zero elements above the 1e-5 gate.

## Demo run (seed 3, 120 OTUs × 24 samples, 5 planted modules, noise 0.3)

- **Automatic RMT threshold: not estimable on this draw.** The scan reports `st=NaN`, `converged=false`, `fallback=true` — 14 candidate thresholds, all rejected (`rejected_gof` from 0.30 to 0.65, then `excluded_tiny_giant`/`excluded_min_nodes`), so under the calibrated eligibility floors none passes the Poisson GOF. The run writes only `report.json` and the `rmt_scan.csv` audit trail: no adjacency, no network, no Fig. 5. Use `--st` to force a fixed threshold.
- **Fixed threshold (`--st 0.52`) on the same seed-3 data:** n=117 kept of 120, E=582, avgK=9.95, GD=3.32, avgCC=0.584, M=0.681, 9 modules (sizes 25/21/15/14/12/10/10/8/2).
- Fig. 5 heatmap: the planted trait–module signal is recovered on fixed-threshold runs.
- Mantel (connectivity vs trait significance, `--st 0.52`): r=0.003, p=0.477 (999 perms) — no global association in this draw, as expected for background-dominated connectivity.

## Benchmarks — cachyos-x8664 desktop via Tailscale (ssh `desktop`, 100.104.77.4)

Hardware: **Ryzen 7 5800X3D (16 threads rayon CPU) + NVIDIA GeForce RTX 2080 Ti 11 GB (Vulkan/wgpu, driver 615.78.08)**. Release build, min-of-3. Adapter line: `[gpu] using adapter: NVIDIA GeForce RTX 2080 Ti (Vulkan)`.

**CPU pipeline scaling:**

| n OTUs × m | total | dominant stage |
|---|---|---|
| 60×12 | 0.0017 s | — |
| 120×12 | 0.0074 s | — |
| 240×24 | 0.043 s | — |
| 480×24 | 0.155 s | — |
| 960×32 | 1.22 s | RMT |
| 1920×48 | 55.06 s | RMT 55.04 s (sim 0.017 s, topology ~0, eigengenes 0.007 s) |

These supersede the earlier topology-dominated profile, which was measured on the degenerate complete graph; provenance: commit `1e8e439`, min-of-3, artifacts off, 99 permutations. Topology is now ~0 s because the corrected threshold yields sparse graphs (E=290 at n=205), so RMT — eigendecomposing the kept subgraph at every candidate threshold — is the whole cost.

**GPU vs CPU similarity matrix (parity: maxdiff 2.3e-7 … 7.9e-7 at all sizes — PASS ≤1e-6):**

| n×m | CPU rayon | GPU RTX 2080 Ti | CPU/GPU |
|---|---|---|---|
| 240×24 | 0.17 ms | 0.376 s | 2200× slower |
| 960×32 | 1.8 ms | 0.357 s | 200× slower |
| 1920×48 | 10.0 ms | 0.382 s | 38× slower |
| 5000×64 | 92.7 ms | 0.448 s | 4.8× slower |
| 8000×96 | 0.496 s | 0.647 s | 1.3× slower |
| 12000×128 | 1.499 s | 1.523 s | **0.98× — parity, not a win** |

Before the rework the GPU won nowhere in [240, 12000]: the fixed ~0.35 s GPU overhead (adapter init + pipeline compile + transfers) dominated below n≈10000, and at 12000×128 the CPU's 1.499 s against the GPU's 1.523 s was parity. The old 4.71× figure is obsolete because the CPU path was parallelised over 16 rayon threads (≈4.7× faster than its own serial baseline, inferred from the old timing rather than re-timed), not because the GPU regressed. The old kernel was tiled 8×8 per thread and processed row-chunks so the storage buffer stayed under wgpu's 128 MiB default binding limit (chunked dispatch added after n=8000 panicked on `max_storage_buffer_binding_size`).

**Kernel rework (2026-11-08, uncommitted).** The similarity kernel is now a tiled SGEMM-style dot product — 16×16 output tiles per 16×16 workgroup, `vec4<f32>` shared-memory staging, host-side row scaling folding in the 1/(m−1) denominator — and the adapter/device/pipeline are created once per process instead of per call. Row-chunks still keep every buffer under 128 MiB, but they now share one pipeline and are submitted in a single command buffer (one submit/poll).

| n×m | before, published (bench.csv) | before, committed kernel re-run | after, warm |
|---|---|---|---|
| 4000×64 | — | 0.372 s | **0.06 s** |
| 8000×96 | 0.647 s | 0.566 s | **0.24 s** |
| 12000×128 | 1.523 s | 1.038 s | **0.53 s** |

Warm = second and later calls in one process; the first call still pays ~0.15–0.25 s of Vulkan device setup, so a one-shot `--gpu` run at 12000 lands around 0.75 s (vs 1.5 s before). Parity is unchanged-to-better: max |GPU−CPU| ≤ 3.84e-7 across parity_sizes (n=2048…3000, m=8), parity_3000 (n=480…3000, m=64) and the big-size run (m=96/128), no element above the 1e-5 gate; ragged shapes with m=5,6,7,13 and constant rows also pass with maxdiff ≤1.4e-7 and no NaN. Measured on the desktop RTX 2080 Ti, release build, 2026-11-08.

CSV: `benches_review/bench.csv` + `bench.log`/`build.log` (commit `1e8e439`) and the earlier `benches_desktop/bench.csv`; plots: `benches_desktop/bench.png`.

## Artifacts per run (`<out-dir>/`)

`adjacency.csv`, `edges.csv`, `nodes.csv` (module, role, zi, Pi, degree, MM, strongest trait), `rmt_scan.csv` (audit trail: every candidate threshold with its status plus the `brody_beta` and `ks_goe` spectral diagnostics), `spectrum.csv`, `degree_dist.csv`, `modules.csv`, `eigengenes.csv`, `fig5_r.csv`, `fig5_p.csv`, `fig5_q.csv`, and `report.json` (topology + RMT scan + timings; with `--null-samples N` it also carries a `null` object with `modularity_z`, `avg_cc_z`, `gd_z`). `similarity.csv` is written only with `--write-similarity`; the Fig-5 heatmap PNG/SVG comes from `plot.py`/`viz.py`. `roles.csv` is no longer written — `nodes.csv` carries the per-OTU role/zi/Pi columns.

## Source paper figures vs this tool

The GitHub repo (`RaphaelRibes/...`) holds the *internship report's* own figures (PCA/PERMANOVA fig 2, Veech co-occurrence fig 3, SRB/DB dynamics fig 4, network figs 5a–c). This CLI implements the **Deng et al. 2012 methodology** that the PDF specifies — including its Figure 5 (eigengene × trait heatmap), which is the graph requested.
