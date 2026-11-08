#import "@preview/cetz:0.5.2": canvas, draw

#set page(paper: "a4", margin: (x: 1.9cm, y: 2.0cm), numbering: "1")
#set text(size: 10pt, font: "Libertinus Serif", lang: "en")
#set par(justify: true, leading: 0.62em)
#set heading(numbering: "1.1")

#show heading.where(level: 1): it => block(above: 1.1em, below: 0.6em)[
  #set text(size: 14pt, weight: "bold")
  #it
]
#show figure.caption: it => block(width: 100%)[
  #set text(size: 8.5pt, style: "italic")
  #it
]

// ---------------------------------------------------------------- CeTZ helpers
#let node(cx, cy, w, h, label, fill, fs: 7pt) = {
  draw.rect((cx - w/2, cy - h/2), (cx + w/2, cy + h/2),
    radius: 0.1, fill: fill, stroke: 0.4pt + rgb("#33414d"))
  draw.content((cx, cy), text(size: fs, fill: rgb("#0d1116"))[#label])
}
#let arrow(a, b) = draw.line(a, b, mark: (end: ">"), stroke: 0.6pt + rgb("#5a6b7a"))

// ==================================================================== title
#align(center)[
  #text(size: 19pt, weight: "bold")[hydro-mena: audit, correction and calibration]
  #v(0.2em)
  #text(size: 10.5pt, style: "italic")[
    A Rust implementation of RMT-based molecular ecological network analysis —
    what was broken, what was fixed, what remains undecided
  ]
  #v(0.3em)
  #text(size: 9pt)[
    Commit `85dc991` · 2026-11-08 · bench on `cachyos-x8664` (Ryzen 7 5800X3D, RTX 2080 Ti)
  ]
]

#v(0.6em)
#block(stroke: 0.5pt + rgb("#b08968"), fill: rgb("#fdf6ec"), inset: 8pt, radius: 3pt)[
  *Scope and honesty statement.* This report documents an audit of an existing
  tool, the defects found, and the evidence for each fix. Where a claim could not
  be verified from a primary source on disk, it is marked as withdrawn rather
  than repeated. The automatic RMT threshold is *not* settled by this work, and
  every threshold-dependent number below is marked exploratory.
]

= Context

`hydro-mena` re-implements the MENA pipeline of Deng et al. (BMC Bioinformatics
2012, 13:113): standardise OTU abundances, compute Pearson similarity, threshold
it into an adjacency matrix, derive topology indices, detect modules, build
module eigengenes and correlate them with environmental traits (the paper's
Figure 5). The implementation is Rust (rayon CPU, optional wgpu GPU) with Python
used only for plotting.

The tool was inherited mid-project with reported results. The audit's job was to
establish whether those results were real before they were used.

= Architecture

#figure(
  canvas({
    let y = 2.4
    node(1.9, y, 3.0, 1.0, [Inputs\ abundance + traits], rgb("#dbe7f3"))
    node(5.6, y, 3.0, 1.0, [Compute core\ Rust + rayon], rgb("#e8f0d8"))
    node(9.3, y, 3.0, 1.0, [Artifacts\ 11 CSV/JSON], rgb("#f6e7d0"))
    node(13.0, y, 3.0, 1.0, [Consumers\ viz + GUI], rgb("#e7dcf0"))
    arrow((3.4, y), (4.1, y))
    arrow((7.1, y), (7.8, y))
    arrow((10.8, y), (11.5, y))
    draw.content((13.0, 1.5), text(size: 6.5pt, fill: rgb("#5a6b7a"))[optional GPU path])
    draw.line((5.6, 1.9), (5.6, 1.5), stroke: 0.4pt + rgb("#9aa8b5"))
    draw.content((5.6, 1.25), text(size: 6.5pt, fill: rgb("#5a6b7a"))[wgpu · f32 · feature-gated])
  }),
  caption: [System boundary. The compute core is the only component that touches numbers;
    the plotting layer and GUI read artifacts and can never disagree with the CLI.]
)

= Pipeline and where it failed

#figure(
  canvas({
    let y = 3.1
    node(1.7, y, 2.5, 0.9, [standardise\ rows], rgb("#dbe7f3"), fs: 6.5pt)
    node(4.5, y, 2.5, 0.9, [similarity\ abs r], rgb("#dbe7f3"), fs: 6.5pt)
    node(7.3, y, 2.5, 0.9, [RMT\ threshold], rgb("#f7d9d9"), fs: 6.5pt)
    node(10.1, y, 2.5, 0.9, [topology\ M · CC · GD], rgb("#e8f0d8"), fs: 6.5pt)
    node(12.9, y, 2.5, 0.9, [modules\ zi · Pi], rgb("#e8f0d8"), fs: 6.5pt)
    node(15.7, y, 2.5, 0.9, [eigengenes\ Fig. 5], rgb("#f6e7d0"), fs: 6.5pt)
    arrow((2.95, y), (3.25, y)); arrow((5.75, y), (6.05, y)); arrow((8.55, y), (8.85, y))
    arrow((11.35, y), (11.65, y)); arrow((14.15, y), (14.45, y))
    draw.content((7.3, 2.25), text(size: 6.5pt, fill: rgb("#a33"))[defective])
    draw.content((10.1, 2.25), text(size: 6.5pt, fill: rgb("#a33"))[inflated Q])
  }),
  caption: [Pipeline stages. The audit found the RMT stage accepted every graph
    (red) and modularity was inflated (red); both contaminated every downstream number.]
)

= Defects found, and the evidence for each

The two scientifically consequential defects share a symptom: they are silent.
Neither crashes, neither warns, and both produce plausible-looking output.

== The RMT stage was a no-op

Unfolded spacings were derived from rank alone:

```rust
let e: Vec<f64> = (0..p).map(|i| (i as f64 + 0.5) / p as f64).collect();
d.push(p as f64 * (e[i + 1] - e[i]));   // == 1.0 for every i
```

Every spacing was exactly 1, so the χ² statistic — itself just `sum((d-1)²)` —
was exactly 0 for *any* graph. The scan therefore accepted its first candidate.
With the start moved to 0.01 in a previous session, that candidate was the
complete graph.

#table(
  columns: (auto, auto, auto),
  stroke: 0.4pt + rgb("#ccd3da"),
  inset: 5pt,
  [*quantity*], [*before*], [*after*],
  [threshold st], [0.010], [0.890],
  [nodes / edges], [200 / 19 500 (complete)], [75 / 105],
  [mean degree], [195.0], [2.8],
  [modularity M], [0.009], [0.683],
  [power-law γ], [−42.4], [1.09],
  [node roles], [200 × "connector"], [mixed],
)
#figure.caption[200 simulated OTUs, default `run`. A negative γ and 100 % connectors
  are the signature of a saturated graph feeding every later stage.]

== Modularity omitted the configuration term

Modularity summed `1 − k_i k_j / 2E` over *existing edges only*. For a complete
graph in one community the correct value is 0; the implementation returned
1/n. Corrected to the aggregate form
$Q = sum_c (L_c / 2E - (d_c / 2E)^2)$, with known-answer tests, every reported
modularity fell. The `ox` condition went from 0.705 to 0.650.

== Other defects

#table(
  columns: (0.5fr, 2.2fr, 2.2fr),
  stroke: 0.4pt + rgb("#ccd3da"), inset: 4.5pt, align: (left, left, left),
  [*No.*], [*defect*], [*status*],
  [1], [GPU matrix computed, then discarded and recomputed on CPU while reporting `gpu_used = true`], [fixed],
  [2], [trait table missing a sample silently returned fewer rows, misaligning every correlation], [fixed, hard error],
  [3], [Mantel permuted pair vectors instead of node labels, inflating significance], [fixed],
  [4], [Jacobi eigensolver rescanned the matrix per rotation, 200 rotations total], [replaced by implicit QR],
  [5], [clustering O(k³), modularity O(n²) per move, serial O(n²m) similarity], [O(k²), O(E), parallel],
  [6], [benchmark ran a stale copy of the pipeline], [now calls `run_pipeline`],
  [7], [`cargo test` broken on a default checkout by feature-gated examples], [fixed, 18 tests],
)

= Performance

Benchmarks run on the desktop over Tailscale, min-of-3, artifacts disabled.

#figure(
  image("fig_bench.png", width: 100%),
  caption: [Scaling and CPU/GPU ratio. Parity is perfect (maxdiff ≤ 7.9×10⁻⁷) but the
    GPU is *not* faster at any tested size.]
)

#table(
  columns: (auto, auto, auto, auto),
  stroke: 0.4pt + rgb("#ccd3da"), inset: 4.5pt,
  [*n × m*], [*pipeline*], [*CPU similarity*], [*GPU similarity*],
  [240 × 24], [0.043 s], [0.00017 s], [0.376 s],
  [960 × 32], [1.22 s], [0.0018 s], [0.357 s],
  [1920 × 48], [55.06 s], [0.0100 s], [0.382 s],
  [12000 × 128], [—], [1.499 s], [1.523 s (0.98×)],
)

Two findings replace the inherited claims. First, the GPU wins nowhere in this
range: the old "4.71× at n = 12000" is obsolete because the *CPU* path was
parallelised, not because the GPU regressed — the kernel still runs at roughly
1 % of f32 peak, so this is a statement about the WGSL kernel, not the hardware.
Second, the bottleneck moved: at n = 1920, RMT is 55.0 s of the 55.06 s total,
because the corrected threshold yields sparse graphs and the eigendecompositions
dominate. Topology is now essentially free.

= The threshold problem

== What the paper specifies

Quoted from `paper.txt` with locations, because an earlier draft of this work
misquoted it:

- *Eq. 3*, the adjacency is *weighted*: $a_(i j) = s_(i j)$ if $s_(i j) >= s_t$, else 0.
- *Step (c)*: "Set an initial threshold value, $s_t$ (e.g., 0.3 based on our experiences)."
- *Step (f)*: unfolding by "fitting the original integrated density to a cubic spline or by local density average".
- *Step (h)*: χ² GOF, $H_0$: Poisson $exp(-d)$, at "a significant level of 0.01".
  An earlier build used 0.05 from a misquote; that was a regression and has been reverted.

== What the 2025 internship code did

```python
adj_matrix = [[value if value >= threshold else 0 for value in row] for row in similarity_matrix]
```

Weighted, diagonal zeroed, and the threshold is *passed in* — the Fig. 5 script
hardcodes 0.52. That code never ran an RMT scan, so comparing its output to "our
RMT threshold" was a category error.

== Decision logic

#figure(
  canvas({
    node(4.2, 4.4, 5.6, 1.05, [Layer 1 — faithful reproduction\ weighted · spline unfold · χ² α = 0.01], rgb("#dbe7f3"), fs: 6.5pt)
    node(4.2, 3.0, 5.6, 1.05, [Layer 2 — continuous diagnostic\ Brody β by MLE + KS vs Poisson and GOE], rgb("#e8f0d8"), fs: 6.5pt)
    node(4.2, 1.6, 5.6, 1.05, [Layer 3 — policy from calibration\ chosen on simulated data, applied once], rgb("#f6e7d0"), fs: 6.5pt)
    arrow((4.2, 3.85), (4.2, 3.55))
    arrow((4.2, 2.45), (4.2, 2.15))
    draw.content((9.6, 4.4), text(size: 6.5pt)[reproduces the method as published])
    draw.content((9.6, 3.0), text(size: 6.5pt)[no knife-edge Boolean, uncertainty reported])
    draw.content((9.6, 1.6), text(size: 6.5pt)[the only layer allowed to pick a number])
  }),
  caption: [Separation of concerns: reproduction, inference, and the policy choice are
    different claims and must not be conflated in one threshold.]
)

== Calibration: does any threshold recover the planted modules?

300 OTUs, 24 samples, 5 planted modules, 10 seeds, 6 thresholds. ARI between
recovered and planted partition (mean ± sd).

#figure(
  image("fig_calibration.png", width: 100%),
  caption: [Recovery falls as the threshold rises, while the RMT procedure scans upward.
    Right panel: the single-level optimiser over-splits — 5 planted modules become 24 at st = 0.70.]
)

#table(
  columns: (auto, auto, auto, auto, auto),
  stroke: 0.4pt + rgb("#ccd3da"), inset: 4.5pt,
  [*st*], [*ARI (noise 0.30)*], [*modules*], [*ARI (noise 0.50)*], [*modules*],
  [0.40], [0.758 ± 0.065], [6.0], [0.233 ± 0.109], [11.6],
  [0.50], [0.692 ± 0.074], [8.3], [0.158 ± 0.043], [18.5],
  [0.55], [0.633 ± 0.100], [12.1], [0.141 ± 0.038], [25.1],
  [0.60], [0.513 ± 0.068], [15.2], [0.141 ± 0.039], [34.0],
  [0.70], [0.447 ± 0.055], [24.3], [0.120 ± 0.039], [46.7],
)

The consequence is uncomfortable and worth stating plainly: recovery is best at
the *lowest* threshold tested, and decays as st rises, whereas the Poisson
criterion walks upward. The criterion optimises spectral randomness, not
community recovery. At noise 0.50 almost nothing is recoverable at any
threshold, so the threshold governs sparsity, not validity.

== Ablation: which guard produces "not estimable"?

Four post-hoc guards were introduced while reading output (giant-component
restriction, Perron trimming, a minimum dof and a minimum spacing count). Each
was toggled independently on the `ox` condition.

#table(
  columns: (auto, auto, auto, auto, auto),
  stroke: 0.4pt + rgb("#ccd3da"), inset: 4.5pt,
  [*configuration*], [*points*], [*st*], [*accepted*], [*kept*],
  [baseline (all guards)], [0], [0.950], [no], [59],
  [no giant component], [0], [0.950], [no], [59],
  [no Perron trim], [0], [0.950], [no], [59],
  [min dof = 0], [1], [0.750], [no], [123],
  [min spacings = 6], [0], [0.950], [no], [59],
  [all guards off], [0], [0.950], [no], [59],
)

Two conclusions, and they separate cleanly:

+ *The scientific verdict is robust.* No configuration accepts a threshold, on
  any of the three conditions. The single point that survives when the dof guard
  is removed (st = 0.75) fails the goodness-of-fit anyway.
+ *The reported point count is not robust.* "Zero estimable points" is produced
  by the dof guard; without it, one point appears. So the phrase "not
  estimable" should be read as "no threshold passes", not "no statistic could be
  computed".

Disabling the giant-component and Perron guards simultaneously empties the
spacing set entirely, which means those two choices are load-bearing for
producing a statistic at all on this data. That is a limitation to state, not a
defect to hide: the unfolding is fragile at $n approx 170$ with 12 to 36 samples.

= Results on the real data (exploratory)

Fixed threshold 0.52 for comparability with the 2025 work, 999 permutations.

#table(
  columns: (auto, auto, auto, auto, auto, auto, auto),
  stroke: 0.4pt + rgb("#ccd3da"), inset: 4.5pt,
  [*condition*], [*n / E*], [*M*], [*avg CC*], [*GD*], [*modules*], [*Mantel r (p)*],
  [box], [149 / 694], [0.631], [0.561], [3.22], [15], [0.173 (0.005)],
  [ox], [166 / 836], [0.650], [0.640], [3.77], [20], [0.125 (0.017)],
  [anox], [165 / 850], [0.583], [0.628], [3.63], [17], [0.077 (0.086)],
)

#figure(
  image("fig_fig5_ox.png", width: 100%),
  caption: [Module eigengene vs trait correlations, ox condition. 20 module rows; the
    12 of 74 traits shown are those with any significant correlation — a display
    filter that must be replaced by q-values or a pre-declared panel.]
)

#grid(
  columns: (1fr, 1fr),
  gutter: 8pt,
  figure(image("fig_network_ox.png", width: 100%), caption: [Co-occurrence network (166 nodes, 836 edges). One disc per module; force-directed layouts were unreadable at this density.]),
  figure(image("fig_roles_ox.png", width: 100%), caption: [Module roles in the zi–Pi plane. Most OTUs are peripherals, consistent with the paper's 90.9 %.]),
)

#grid(
  columns: (1fr, 1fr),
  gutter: 8pt,
  figure(image("fig_rmt_ox.png", width: 100%), caption: [RMT diagnostic. With the faithful weighted matrix and α = 0.01 no threshold is estimable on this data.]),
  figure(image("fig_spectrum_ox.png", width: 100%), caption: [Adjacency spectrum and nearest-neighbour spacing distribution vs Poisson and Wigner.]),
)

#figure(
  image("fig_fig5_box.png", width: 100%),
  caption: [box condition, shown for comparison: fewer modules (15) and the lowest Mantel correlation of the three.]
)

= What remains

The honest status is that the *software* is in good shape and the *science* is
not finished.

+ *Threshold policy*: unresolved. Layer 1 and 2 are specified but not
  implemented; the calibration says the automatic criterion is anti-correlated
  with recovery, so publishing an RMT threshold as "the" threshold is not
  defensible.
+ *Multiple testing*: Benjamini–Hochberg q-values are now computed
  (`fig5_q.csv`), but the display filter still selects traits by raw
  significance, which is cherry-picking under multiplicity. Fix pending.
+ *Null model*: M, avg CC and GD still have no null distribution. Without a
  degree-preserving rewiring null, "M = 0.65 is high" is unquantified.
+ *Over-splitting*: the single-level Louvain optimiser fragments communities
  (5 planted → 24 recovered). Multilevel aggregation is the fix.
+ *Compositionality*: Pearson on relative abundances is biased; CLR-based
  similarity is the standard mitigation and is not implemented.
+ *Provenance*: the vault is not under version control and `/real/` is excluded
  from git, so validation inputs live only on RawStorage.

= Reproducibility

```bash
cargo build --release                     # 18 tests
hydro-mena run --abundance real/abundance_ox.csv \
               --traits real/traits_ox.csv \
               --out-dir real/review_ox/s52 --st 0.52 --permutations 999
pixi run python viz.py all real/review_ox/s52 --out real/review_ox/figs
pixi run python gui/app.py --root . --port 8770
```
