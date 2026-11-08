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

#let node(cx, cy, w, h, label, fill, fs: 7pt) = {
  draw.rect((cx - w/2, cy - h/2), (cx + w/2, cy + h/2),
    radius: 0.1, fill: fill, stroke: 0.4pt + rgb("#33414d"))
  draw.content((cx, cy), text(size: fs, fill: rgb("#0d1116"))[#label])
}
#let arrow(a, b) = draw.line(a, b, mark: (end: ">"), stroke: 0.6pt + rgb("#5a6b7a"))

#align(center)[
  #text(size: 19pt, weight: "bold")[mena: audit, correction and a working threshold]
  #v(0.2em)
  #text(size: 10.5pt, style: "italic")[
    A Rust implementation of RMT-based molecular ecological network analysis —
    what was broken, what was fixed, and what the fixed estimator now says
  ]
  #v(0.3em)
  #text(size: 9pt)[
    Commit `abacc49` · 2026-11-08 · bench on `cachyos-x8664` (Ryzen 7 5800X3D, RTX 2080 Ti)
  ]
]

#v(0.6em)
#block(stroke: 0.5pt + rgb("#b08968"), fill: rgb("#fdf6ec"), inset: 8pt, radius: 3pt)[
  *Scope and honesty statement.* This report documents an audit of an inherited
  tool, and a correction to the audit itself. An earlier version concluded that no
  threshold was estimable on the real data. That conclusion was wrong, for an
  instructive reason: the goodness-of-fit test was never actually executed.
  Section 7 shows how the artefact was found and what the estimator says once it
  works.
]

= Context

`mena` re-implements the MENA pipeline of Deng et al. (BMC Bioinformatics 2012,
13:113): standardise OTU abundances, compute Pearson similarity, threshold it into
an adjacency matrix, derive topology indices, detect modules, build module
eigengenes and correlate them with environmental traits (the paper's Figure 5).
The implementation is Rust (rayon CPU, optional wgpu GPU); Python is used only for
plotting.

The tool was inherited with reported results. The audit's job was to establish
whether those results were real before they were used.

= Architecture

#figure(
  canvas({
    let y = 2.4
    node(1.9, y, 3.0, 1.0, [Inputs\ abundance + traits], rgb("#dbe7f3"))
    node(5.6, y, 3.0, 1.0, [Compute core\ Rust + rayon], rgb("#e8f0d8"))
    node(9.3, y, 3.0, 1.0, [Artifacts\ 10 CSV/JSON], rgb("#f6e7d0"))
    node(13.0, y, 3.0, 1.0, [Consumers\ viz + GUI], rgb("#e7dcf0"))
    arrow((3.4, y), (4.1, y))
    arrow((7.1, y), (7.8, y))
    arrow((10.8, y), (11.5, y))
    draw.line((5.6, 1.9), (5.6, 1.5), stroke: 0.4pt + rgb("#9aa8b5"))
    draw.content((5.6, 1.25), text(size: 6.5pt, fill: rgb("#5a6b7a"))[wgpu · f32 · feature-gated])
  }),
  caption: [System boundary. The compute core is the only component that touches
    numbers; the plotting layer and GUI read artifacts and cannot disagree with the CLI.]
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
    draw.content((7.3, 2.25), text(size: 6.5pt, fill: rgb("#a33"))[two defects])
    draw.content((10.1, 2.25), text(size: 6.5pt, fill: rgb("#a33"))[inflated Q])
  }),
  caption: [Pipeline stages. The RMT stage carried two independent defects (red)
    and modularity was inflated (red); both contaminated every downstream number.]
)

= Defects found

== The RMT stage was first a no-op, then blind

Unfolded spacings were originally derived from rank alone:

```rust
let e: Vec<f64> = (0..p).map(|i| (i as f64 + 0.5) / p as f64).collect();
d.push(p as f64 * (e[i + 1] - e[i]));   // == 1.0 for every i
```

Every spacing was exactly 1, so the χ² statistic was exactly 0 for any graph and
the scan accepted its first candidate — the complete graph.

Repairing that revealed a second, quieter defect. The scan was built on
`filter_map`, so a candidate that produced no statistic simply vanished:

```rust
.filter_map(|&st| evaluate(s, st, min_nodes, alpha, cfg))
```

Four rejection reasons collapsed into "this threshold does not exist". With an
empty `rmt_scan.csv` the output could not distinguish *the data rejects Poisson*
from *the guards rejected the data*. Section 7 is the story of that distinction.

#table(
  columns: (auto, auto, auto),
  stroke: 0.4pt + rgb("#ccd3da"),
  inset: 5pt,
  [*quantity*], [*before*], [*after*],
  [threshold st], [0.010], [0.500 – 0.600],
  [nodes / edges], [200 / 19 500 (complete)], [149 / 758],
  [mean degree], [195.0], [10.2],
  [modularity M], [0.009], [0.591],
  [power-law γ], [−42.4], [−1.9],
  [node roles], [200 × "connector"], [mixed],
)
#figure.caption[200 simulated OTUs, default `run`. A γ of −42 and 100 % connectors
  are the signature of a saturated graph feeding every later stage.]

== Modularity omitted the configuration term

Modularity summed `1 − k_i k_j / 2E` over existing edges only. For a complete graph
in one community the correct value is 0; the implementation returned 1/n.
Corrected to $Q = sum_c (L_c / 2E - (d_c / 2E)^2)$ with known-answer tests. Every
reported modularity fell; `ox` went from 0.705 to 0.650.

== Other defects

#table(
  columns: (0.4fr, 2.3fr, 2.2fr),
  stroke: 0.4pt + rgb("#ccd3da"), inset: 4.5pt, align: (left, left, left),
  [*No.*], [*defect*], [*status*],
  [1], [GPU matrix computed, then discarded and recomputed on CPU while reporting `gpu_used = true`], [fixed],
  [2], [trait table missing a sample silently returned fewer rows, misaligning every correlation], [fixed, hard error],
  [3], [Mantel permuted pair vectors instead of node labels, inflating significance], [fixed],
  [4], [Jacobi eigensolver rescanned the matrix per rotation, 200 rotations total], [replaced by implicit QR],
  [5], [clustering O(k³), modularity O(n²) per move, serial O(n²m) similarity], [O(k²), O(E), parallel],
  [6], [benchmark ran a stale copy of the pipeline], [now calls `run_pipeline`],
  [7], [`cargo test` broken on a default checkout by feature-gated examples], [fixed, 18 tests],
  [8], [Fig-5 panel selected traits by significance, i.e. cherry-picking under multiplicity], [ranks by effect size],
  [9], [commits landed with a broken test build (twice)], [pre-commit build+test gate],
)

= Performance

Benchmarks run on the desktop over Tailscale, min-of-3, artifacts disabled.

#figure(
  image("fig_bench.png", width: 100%),
  caption: [Scaling and CPU/GPU ratio. Parity is exact (maxdiff ≤ 7.9×10⁻⁷) but the
    GPU is not faster at any tested size.]
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

The inherited claim of a 4.71× GPU win at n = 12000 is obsolete because the CPU
path was parallelised, not because the GPU regressed: the kernel still runs at
roughly 1 % of f32 peak, so this is a statement about the WGSL kernel, not the
hardware. The bottleneck also moved: at n = 1920 the RMT stage was 55.0 s of the
55.06 s total. After the unfolding fix the three-condition real-data sweep runs in
under a second, because unfolding is now linear rather than a badly conditioned
polynomial fit.

= The threshold problem

== What the paper specifies

- *Eq. 3*: the adjacency is weighted, $a_(i j) = s_(i j)$ if $s_(i j) >= s_t$, else 0.
- *Step (c)*: "Set an initial threshold value, $s_t$ (e.g., 0.3 based on our experiences)."
- *Step (f)*: unfold by "fitting the original integrated density to a cubic spline or by local density average".
- *Step (h)*: χ² GOF, $H_0$: Poisson $exp(-d)$, at "a significant level of 0.01".

== What the 2025 internship code did

```python
adj_matrix = [[value if value >= threshold else 0 for value in row] for row in similarity_matrix]
```

Weighted, diagonal zeroed, threshold passed in — the Fig. 5 script hardcodes 0.52.
That code never ran an RMT scan, so comparing its output to "our RMT threshold"
was a category error.

== Decision logic

#figure(
  canvas({
    node(4.2, 4.4, 6.2, 1.05, [Layer 1 — faithful reproduction\ weighted · local-density unfold · χ² α = 0.01], rgb("#dbe7f3"), fs: 6.5pt)
    node(4.2, 2.9, 6.2, 1.05, [Layer 2 — continuous diagnostic\ Brody β by MLE + KS vs Poisson and GOE], rgb("#e8f0d8"), fs: 6.5pt)
    node(4.2, 1.4, 6.2, 1.05, [Layer 3 — policy from calibration\ chosen on simulated data, applied once], rgb("#f6e7d0"), fs: 6.5pt)
    arrow((4.2, 3.85), (4.2, 3.45))
    arrow((4.2, 2.35), (4.2, 1.95))
    draw.content((10.2, 4.4), text(size: 6.5pt)[reproduces the published method])
    draw.content((10.2, 2.9), text(size: 6.5pt)[no knife-edge Boolean])
    draw.content((10.2, 1.4), text(size: 6.5pt)[the only layer allowed to pick a number])
  }),
  caption: [Layer 1 is now implemented and returns a threshold on two of three
    conditions. Layers 2 and 3 remain proposals.]
)

== Calibration: which threshold recovers planted structure?

300 OTUs, 24 samples, 5 planted modules, 10 seeds.

#figure(
  image("fig_calibration.png", width: 100%),
  caption: [Recovery falls as the threshold rises, while the RMT criterion scans
    upward. Right panel: the single-level optimiser over-splits — 5 planted modules
    become 24 at st = 0.70.]
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

Recovery is best at the lowest threshold tested and decays as st rises, whereas the
Poisson criterion walks upward. The criterion optimises spectral randomness, not
community recovery. This is a tension to state, not to resolve by preference.

== Literature anchors (Europe PMC, 2026-11-08)

Searches for a dedicated critique of RMT threshold selection in MENA returned very
little: both hits are tool papers. The criterion propagates by inheritance.

+ Peng X, Feng K, Yang X, He Q, Zhao B, Li T, Wang S, *Deng Y.* (2024). iNAP 2.0:
  harnessing metabolic complementarity in microbial network analysis. *iMeta*
  3:e235. doi:10.1002/imt2.235 — the original authors' successor implementation,
  and the fair comparison target.
+ Zhou M, Xu Y, Zhu K et al. (2025). Spatial-temporal pattern of RMT network
  stability in an estuarine phytoplankton community. *J Environ Manage*.
  doi:10.1016/j.jenvman.2025.126873 — uses the RMT threshold as a stability
  diagnostic rather than a selector.
+ Méndez-Bermúdez J A et al. (2015). Universality in the spectral and eigenfunction
  properties of random networks. *Phys Rev E* 91:032122.

= Fixing the estimator

== The unfolding was the artefact

The first repair kept the polynomial fit with monotone clipping: space the
eigenvalues by a degree-5 fit to the integrated density and clamp any decrease.
Clipping creates plateaus, plateaus create zero spacings, and the code rejected any
spectrum containing one. The audit trail made this visible — at every threshold, on
every condition:

```
9  excluded_too_few_spacings
4  excluded_tiny_giant
1  excluded_low_dof
0  rejected_gof
```

Zero. The goodness-of-fit had never run. The audit's conclusion that "no threshold
is estimable" was a statement about the preprocessing, not about the data. The
distinction matters: the first claim is a finding, the second is a bug.

#figure(
  canvas({
    node(2.7, 3.2, 4.0, 2.0, [], rgb("#f4f7fa"), fs: 6pt)
    node(2.7, 3.88, 3.4, 0.40, [spectrum $lambda_i$], rgb("#dbe7f3"), fs: 7pt)
    draw.line((1.2, 2.55), (1.6, 2.95), (1.9, 2.62), (2.3, 3.15), (2.7, 2.75), (3.1, 3.30), (3.5, 3.00), (3.9, 3.45), stroke: 0.7pt + rgb("#33414d"))
    node(7.3, 3.2, 4.0, 2.0, [], rgb("#f4f7fa"), fs: 6pt)
    node(7.3, 3.88, 3.4, 0.40, [integrated density, local window], rgb("#e8f0d8"), fs: 7pt)
    draw.line((5.9, 2.60), (6.4, 2.75), (6.9, 3.00), (7.4, 3.30), (7.9, 3.50), (8.4, 3.62), stroke: 0.7pt + rgb("#33414d"))
    draw.line((7.3, 2.45), (7.3, 3.70), stroke: 0.5pt + rgb("#a33"), dash: "dashed")
    node(11.9, 3.2, 4.0, 2.0, [], rgb("#f4f7fa"), fs: 6pt)
    node(11.9, 3.88, 3.4, 0.40, [spacings $d_i$, all positive], rgb("#f6e7d0"), fs: 7pt)
    draw.line((10.5, 2.55), (10.9, 3.10), (11.2, 2.70), (11.5, 3.25), (11.8, 2.85), (12.1, 3.20), (12.4, 2.90), (12.7, 3.30), stroke: 0.7pt + rgb("#33414d"))
    arrow((4.75, 3.2), (5.15, 3.2))
    arrow((9.35, 3.2), (9.75, 3.2))
    draw.content((7.3, 1.45), text(size: 6.5pt, fill: rgb("#5a6b7a"))[
      polynomial + clipping gives plateaus → zero spacings → the whole spectrum was rejected
    ])
  }),
  caption: [Replacing the clipped polynomial with a local density average. Spacings
    are positive by construction, so no candidate is discarded before the test runs.]
)

== Eligibility floors, from the calibration

The old floors were `n/10` nodes, chosen while reading output. They are now two
config values with a stated basis: at least 50 % of OTUs retained and at least 50 %
of kept nodes in the giant component. The calibration pilot shows ARI falling below
0.35 beneath 50 % retention, so a sparser graph is not a usable estimate of
community structure.

== What the fixed estimator reports

#figure(
  image("fig_scan_audit.png", width: 100%),
  caption: [The audit trail, one point per candidate. Above the dashed line χ²
    exceeds its critical value. Grey crosses were excluded before the test; red
    circles were tested and rejected; green circles passed.]
)

#table(
  columns: (auto, auto, auto, auto, auto),
  stroke: 0.4pt + rgb("#ccd3da"), inset: 4.5pt,
  [*condition*], [*accepted*], [*rejected GOF*], [*excluded*], [*candidates*],
  [box], [1], [12], [5], [18],
  [ox], [0], [8], [6], [14],
  [anox], [2], [10], [6], [18],
)

#table(
  columns: (auto, auto, auto, auto, auto, auto, auto),
  stroke: 0.4pt + rgb("#ccd3da"), inset: 4.5pt,
  [*condition*], [*st (auto)*], [*n / E*], [*M*], [*avg CC*], [*GD*], [*modules*],
  [box], [0.500], [149 / 758], [0.591], [0.566], [3.07], [16],
  [anox], [0.600], [159 / 611], [0.688], [0.653], [4.37], [22],
  [ox], [not estimable], [—], [—], [—], [—], [—],
)

Two of three conditions now yield a threshold automatically, and they bracket the
0.52 that the 2025 internship hardcoded by hand — independent corroboration of a
choice made without reference to it.

`ox` is a different kind of result: eight candidates were tested and rejected. Its
spacing distribution does not reach Poisson within the eligible range. That is now
a scientific statement, because the test actually ran.

= Results at a fixed threshold

Fixed 0.52 for comparability with the 2025 work, 999 permutations. These are the
reproduction numbers; they are not the automatic path.

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
  caption: [Module eigengene vs trait correlations, ox. Panels are now chosen by
    effect size, not by significance.]
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
  figure(image("fig_fig5_box.png", width: 100%), caption: [box condition: fewer modules (15) and the lowest Mantel correlation of the three.]),
  figure(image("fig_spectrum_ox.png", width: 100%), caption: [Adjacency spectrum and nearest-neighbour spacings against Poisson and Wigner.]),
)

= What remains

The software is in defensible shape. The science is half-finished, and the half
that is finished is the half that can be automated.

+ *Layers 2 and 3*: Brody β and the calibration-derived policy are specified but not
  implemented. Only Layer 1 exists, and it fails on one of three conditions.
+ *ox*: non-estimable. Whether that is biology (a spectrum genuinely closer to
  Poisson) or sample size (12 samples, 166 OTUs) is unresolved, and guessing would
  repeat the error this report documents.
+ *Calibration vs criterion*: recovery is best at low thresholds while the criterion
  selects higher ones. Publishing an automatic threshold without acknowledging this
  trade-off would be dishonest.
+ *Null models*: M, avg CC and GD still have no degree-preserving null, so
  "M = 0.65 is high" remains unquantified.
+ *Over-splitting*: the single-level optimiser fragments communities (5 planted → 24
  recovered). Multilevel aggregation is the fix.
+ *Compositionality*: Pearson on relative abundances is biased; CLR-based similarity
  is not implemented.
+ *Comparison*: installing iNAP 2.0 and running it on the same three inputs would
  settle the modularity gap far better than argument.

= Reproducibility

```bash
cargo build --release                              # 18 tests, pre-commit gate
mena run --abundance real/abundance_ox.csv --traits real/traits_ox.csv \
         --out-dir real/review_ox/auto              # automatic (RMT) threshold
mena run --abundance real/abundance_ox.csv --traits real/traits_ox.csv \
         --out-dir real/review_ox/s52 --st 0.52     # forced threshold
pixi run python viz.py all real/review_ox/s52 --out real/review_ox/figs
pixi run python gui/app.py --root . --port 8770
```

Every run writes `rmt_scan.csv` with one row per candidate and an explicit `status`.
If no candidate is accepted, no adjacency is written at all: the tool reports
`not estimable` rather than inventing a graph.
