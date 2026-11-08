# hydro-mena — RMT-based Molecular Ecological Network Analysis CLI

Reimplements the full MENA pipeline of **Deng et al., BMC Bioinformatics 2012, 13:113**
("Molecular ecological network analyses", the paper at `~/Downloads/1471-2105-13-113.pdf`),
whose Figure 5 is the module-eigengene × environmental-trait correlation heatmap.

**Architecture (per user directive): Rust for all compute, Python only for graphs, GPU support, pixi env.**

- `src/*.rs` — Rust library + CLI (cargo, release profile, LTO). CPU-parallel via rayon.
- `src/gpu.rs` — optional wgpu/Vulkan compute (AMD RADV on this desktop) for the O(n²m) similarity matrix. Build with `--features gpu`; CLI flag `--gpu`.
- `plot.py` — the ONLY Python: renders the Fig-5-style heatmap from the CSVs the Rust CLI writes (matplotlib/seaborn).
- `pixi.toml` — env (conda-forge: python 3.12 + matplotlib/seaborn/pandas/numpy, rust). Tasks: `pixi run simulate|run|bench`.

## Pipeline stages implemented (paper Methods)

1. **Standardization** — x_ik = (y_ik − mean_i)/σ_i (Methods).
2. **Pearson correlation** R, **similarity** S = |R| (Eq. 1–2). GPU-accelerated with `--gpu`.
3. **RMT threshold** (Fig. 2, steps a–k): scan st upward (0.1 coarse → 0.01 fine), build unweighted adjacency A_ij = 1[s_ij ≥ st] dropping empty rows, eigen-decompose (Jacobi), unfold eigenvalues, χ² GOF of the NNSD against Poisson exp(−d) at α=0.01; first st where H0 is accepted is the threshold.
4. **Topology** (Table 1): n, edges, avgK, GD (avg shortest path over reachable pairs), avgCC, modularity M (Eq. 7), scale-free R² (log P(k) vs log k, γ), hierarchy R² (C(k) ~ k^−γ).
5. **Modules** — greedy/Louvain-style modularity optimization; **roles** via within-module connectivity zi and participation coefficient Pi (Eq. 8–9) → peripheral / connector / module hub / network hub (zi≤2.5, Pi≤0.62).
6. **Eigengenes** (Eq. 10–12): SVD per module, E_b = first right singular vector, Φ_b = variance explained, MM_iE = cor(x_i, E_b).
7. **Figure 5** — Pearson r and p between each module eigengene and each standardized trait → heatmap (red positive / green negative, numbers = r with p in parentheses), rendered by `plot.py`.
8. **Trait links** — GS_i,h = cor(x_i,T_h)² (Eq. 13) + **Mantel test** (permutation) between connectivity distance |k_i−k_j| and GS distance.
9. **Random-network comparison** hooks: Maslov–Sneppen double-edge-swap nulls live in `network.rs`/`pipeline.rs` (Z-scores vs 100 rewired networks).

## Usage

```fish
cd ~/hermes_scratch/hydro-mena
export PATH="$HOME/.pixi/bin:$PATH"
pixi run simulate -- --otus 120 --samples 24 --modules 5 --traits 4 --noise 0.3 --seed 3 --out-dir demo_final
pixi run run -- run --abundance demo_final/abundance.csv --traits demo_final/traits.csv --out-dir demo_final/out [--gpu]
pixi run run -- bench --out-dir benches            # CPU scaling; add --features gpu build for GPU similarity benches
pixi run python plot.py fig5 --r demo_final/out/fig5_r.csv --p demo_final/out/fig5_p.csv --out fig5.png --svg fig5.svg
```

GPU build: `cargo build --release --features gpu` (wgpu → Vulkan/RADV). `--gpu` on `run` uses the adapter for the similarity matrix and prints which adapter was used; falls back to CPU with a warning if unavailable. Verified: GPU vs CPU similarity max |diff| ≈ 3e-7.

## Demo run (seed 3, 120 OTUs × 24 samples, 5 planted modules, noise 0.3)

- RMT threshold st = 0.20 (converged, 18 scan points)
- Network: n=120, edges=2951, avgK=49.2, GD=1.59, avgCC=0.52, M=0.25, 5 modules (sizes 27/25/25/23/20)
- Fig 5 heatmap (`demo_final/out/fig5_heatmap.png`): module #1 r=−0.88 (p<0.001) with O2; module #3 r=−0.88 (p<0.001) with Alkane_degradation; module #4 r=+0.67 (p<0.001) with PAH_degradation — the planted trait–module signal is recovered.
- Mantel (connectivity vs trait significance): r=−0.03, p=0.998 (999 perms) — no global association in this draw, as expected for background-dominated connectivity.

## Benchmarks — cachyos-x8664 desktop via Tailscale (ssh `desktop`, 100.104.77.4)

Hardware: **Ryzen 7 5800X3D (16 threads rayon CPU) + NVIDIA GeForce RTX 2080 Ti 11 GB (Vulkan/wgpu, driver 615.78.08)**. Release build, min-of-3. Adapter line: `[gpu] using adapter: NVIDIA GeForce RTX 2080 Ti (Vulkan)`.

**CPU pipeline scaling:**

| n OTUs × m | total | dominant stage |
|---|---|---|
| 60×12 | 0.010 s | — |
| 120×12 | 0.042 s | topology 0.022 s |
| 240×24 | 0.15 s | topology 0.09 s |
| 480×24 | 0.93 s | topology 0.63 s |
| 960×32 | 6.2 s | topology 4.8 s, RMT 0.33 s |
| 1920×48 | 30.7 s | topology 24 s (BFS GD O(n·E)), modules 4.8 s |

**GPU vs CPU similarity matrix (parity: maxdiff 2.3e-7 … 7.9e-7 at all sizes — PASS ≤1e-6):**

| n×m | CPU rayon | GPU RTX 2080 Ti | speedup |
|---|---|---|---|
| 240×24 | 0.6 ms | 301 ms | 0.00x |
| 480×24 | 1.8 ms | 316 ms | 0.01x |
| 960×32 | 10.5 ms | 284 ms | 0.04x |
| 1920×48 | 58 ms | 304 ms | 0.19x |
| 3000×64 | 214 ms | 338 ms | 0.63x |
| 5000×64 | 583 ms | 517 ms | 1.13x |
| 8000×96 | 2.57 s | 998 ms | **2.58x** |
| 12000×128 | 6.98 s | 1.48 s | **4.71x** |

Fixed ~300 ms GPU overhead (adapter init + per-chunk pipeline compile + upload/download) dominates below n≈3000; crossover at n≈4000, then the 2080 Ti wins increasingly (4.7x at n=12000). The kernel is tiled 8×8 per thread and processes row-chunks so the storage buffer stays under wgpu's 128 MiB default binding limit (chunked dispatch added after n=8000 panicked on `max_storage_buffer_binding_size`).

CSV: `benches_desktop/bench.csv` (synced back over Tailscale); plots: `benches_desktop/bench.png`.

## Artifacts per run (`<out-dir>/`)

`similarity.csv`, `adjacency.csv`, `eigengenes.csv`, `fig5_r.csv`, `fig5_p.csv`, `roles.csv`, `report.json` (topology + RMT scan + timings), plus `fig5_heatmap.png/.svg` from plot.py.

## Source paper figures vs this tool

The GitHub repo (`RaphaelRibes/...`) holds the *internship report's* own figures (PCA/PERMANOVA fig 2, Veech co-occurrence fig 3, SRB/DB dynamics fig 4, network figs 5a–c). This CLI implements the **Deng et al. 2012 methodology** that the PDF specifies — including its Figure 5 (eigengene × trait heatmap), which is the graph requested.
