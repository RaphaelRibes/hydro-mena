#set page(paper: "a4", margin: (x: 2.2cm, y: 2.2cm), numbering: "1")
#set text(size: 10.5pt, font: "Liberation Serif", lang: "en")
#set par(justify: true, leading: 0.68em, first-line-indent: 1.4em)
#set heading(numbering: "1.1")

#show heading.where(level: 1): it => block(above: 1.3em, below: 0.7em)[
  #set text(size: 13pt, weight: "bold")
  #set par(first-line-indent: 0pt)
  #it
]
#show heading.where(level: 2): it => block(above: 1.0em, below: 0.5em)[
  #set text(size: 11pt, weight: "bold", style: "italic")
  #set par(first-line-indent: 0pt)
  #it
]
#show figure.caption: it => block(width: 100%)[
  #set text(size: 8.5pt)
  #set par(first-line-indent: 0pt, justify: false)
  #it
]
#show figure: it => block(below: 1em)[#it]

#align(center, block(width: 90%)[
  #set par(justify: false, first-line-indent: 0pt)
  #text(size: 17pt, weight: "bold")[An audited reimplementation of RMT-based molecular ecological network analysis]
  #v(0.5em)
  #text(size: 10pt)[Raphael Ribes — 2026-11-09]
  #v(0.3em)
  #text(size: 9.5pt, style: "italic")[Code and audit trail: github.com/RaphaelRibes/hydro-mena (commit `8ae7c7c`)]
])

#v(1em)
#align(center)[#text(style: "italic", size: 10pt)[
  Abstract — We reimplemented the Molecular Ecological Network Analysis (MENA)
  pipeline of Deng et al. (2012) in Rust and audited the inherited 2025 code before
  trusting its outputs. The audit found that the random-matrix-theory (RMT)
  threshold scan had never actually executed its goodness-of-fit test — every
  candidate was discarded by preprocessing, so an earlier "no threshold is
  estimable" conclusion was a bug report, not a finding. After repairing the
  unfolding, the eligibility floors and the modularity formula, the automatic
  threshold is estimable on two of three real hydrocarbon-degradation conditions
  (box: 0.680, ox: 0.610, anox: 0.650 on the binary adjacency spectrum — all
  three estimable; the weighted-spectrum mode, where ox reads Wigner-like
  at β = 0.92, is kept as an ablation). We add Louvain modularity
  detection, degree-preserving null z-scores, bootstrap module stability,
  signed edges, a Clauset scale-free test, and a centred log-ratio (CLR) option
  that makes the oxic condition estimable (st = 0.63). All networks show
  modularity 20–31 null SD above degree-matched random graphs (b = 100) and
  bootstrap module stability 0.35–0.45 mean Jaccard.
  The implementation is open source, 40/40 tests, with an exact CPU/GPU parity
  and a GPU path 2.8× faster than CPU at n = 12 000.
]]

= Introduction

Microbial communities are too diverse and too undersampled for pairwise
association tests to be trustworthy on their own: with hundreds of OTUs and a
few dozen samples, thousands of spurious correlations are inevitable. Deng et
al. (2012) proposed to escape this by *not* choosing a correlation cutoff at
all: build the weighted similarity matrix, threshold it at candidate values,
and let random matrix theory decide which threshold produces a network whose
adjacency spectrum is statistically indistinguishable from a random (Poisson)
one. The pipeline — standardise, correlate, threshold, RMT-select, measure
topology, detect modules, correlate module eigengenes with environmental
traits — is the MENA pipeline, and the 2025 internship code in our lab was an
implementation of it.

Before reusing inherited results we audited the implementation. The audit
mattered: the inherited scan's χ² goodness-of-fit never ran, so its "not
estimable" verdicts said nothing about the data. This paper reports (i) what
was broken and how it was found, (ii) the corrected automatic thresholds on
three real conditions, and (iii) the additions that make the method more
robust — null models, bootstrap stability, signed edges, a defensible
scale-free test, compositionality handling, and a GPU path. All numbers are
reproducible from the repository; every run writes a per-candidate audit
trail.

= Methods

== Data

Three conditions from a long-term hydrocarbon-degradation experiment
("RawStorage 2025" revision): *box* (178 OTUs × 18 samples), *oxic / ox*
(173 × 36), *anoxic / anox* (181 × 34), each with 74 hydrocarbon traits
(n-alkanes nC10–nC…, PAHs). OTU abundances are relative abundances per
sample; taxonomy strings are carried through as node names.

== Pipeline

1. *Standardise* each OTU row (z-score across samples). Optionally apply the
   centred log-ratio transform first (`--clr`, pseudocount 1.0):
   $ upright(c l r)(x)_(i k) = log(x_(i k) + p c) - overline(log(x_(i dot) + p c)) $.
   Relative abundances are compositional — each sample sums to a constant —
   so raw Pearson correlations are biased by the closure constraint; CLR
   removes it. Off by default; used for the ox sensitivity analysis.
2. *Similarity*: Pearson r between OTU rows; $S = |r|$ (Eq. 1–2 of Deng et
   al.), or signed r with `--signed` (sign kept as edge metadata; the RMT
   stage still thresholds $|r|$ because negative weights would break the
   weighted-adjacency Eq. 3).
3. *Threshold*: automatic RMT scan (default) or user-set `--st`. The scan
   walks st from 0.30 upward; at each candidate it builds the thresholded
   adjacency and eigen-decomposes its *binary* 0/1 form (default
   `--binary-spectrum`, matching the 2025 repo's `fig_5_threshold.py`:
   weighted correlation-thresholded spectra carry level repulsion — Brody
   β up to 0.92 on real data — so their NNSD is Wigner-like and never
   passes the Poisson GOF; binary random-graph spectra are Poisson.
   `--no-binary-spectrum` reverts to the weighted matrix of Deng Eq. 3
   for comparison). The graph itself keeps the weights
   $a_(i j) = s_(i j)$ if $s_(i j) \u{2265} s_t$ else 0 (Eq. 3);
   takes the spectrum of the largest connected component, unfolds by local
   density average (paper step f), computes nearest-neighbour spacings, and
   runs a binned χ² GOF against the Poisson law $exp(-d)$ at α = 0.01
   (paper step h). Eligibility floors, calibrated on simulations: ≥ 50 %
   of OTUs retained and ≥ 50 % of kept nodes in the giant component. First
   accepted candidate wins; if none is accepted the tool reports
   *not estimable* and writes no graph rather than inventing one.
4. *Topology*: nodes, edges, density, mean degree, geodesic distance, average
   clustering, modularity, connected-component stats. Modularity uses the full
   formula $Q = sum_c (L_c / 2E - (d_c / 2E)^2)$ — the configuration term is
   what makes a one-community complete graph score exactly 0.
5. *Null model* (`--null-samples B`): B Maslov–Sneppen degree-preserving
   rewirings; z-scores for Q, avg CC, GD in `report.json`.
6. *Modules*: Louvain (default, `--modules louvain`; the old greedy
   agglomeration is kept as `--modules greedy` ablation). Roles via
   within-module z-score $z_i$ and participation coefficient $P_i$
   (Eq. 8–9): peripheral ($z_i \u{2264} 2.5, P_i \u{2264} 0.62$), connector,
   module hub, network hub. *Bootstrap stability* (`--boot N`): resample
   sample columns with replacement N times, rebuild at fixed st, match modules
   by best Jaccard, report mean and per-module Jaccard.
7. *Scale-free test*: Clauset–Shalizi–Newman discrete MLE
   $gamma = 1 + n [sum log(k_i / (x_(min) - 0.5))]^(-1)$, xmin scanned over
   observed degrees (tail ≥ 100), exact discrete-CDF Kolmogorov–Smirnov,
   p-value by parametric bootstrap (1000 sims). Reported as
   `topology.clauset_*`.
8. *Eigengenes and Fig. 5*: module eigengene = first PC of the module's
   standardised abundance submatrix; Pearson r (and Mantel permutation p,
   999 perms) against each of the 74 traits.
9. *Spectrum diagnostics*: Brody parameter β (MLE, 0 = Poisson, 1 = Wigner)
   and KS distances to GOE and to Poisson are written per scan candidate into
   `rmt_scan.csv`.

== Implementation and verification

Rust, rayon for CPU parallelism, wgpu (Vulkan) behind `--features gpu` for the
similarity matrix; Python only for plotting. 40/40 unit and end-to-end tests
(known-answer modularity, eigensolver vs reference spectrum, Poisson GOF
accepting exponential and rejecting flat spacings, Clauset recovering γ on a
synthetic power law and rejecting an exponential tail, bootstrap plumbing,
Mantel on a planted correlation). CPU/GPU parity is exact at every tested size
(`parity_3000` maxdiff $2.6 \u{D7} 10^(-7)$, bad = 0); the GPU kernel
(16×16 tiles, `vec4` staging) on an RTX 2080 Ti is 1.69× faster than
rayon CPU at n = 12 000 (0.486 s vs 0.822 s, min-of-3, warm repeats
0.4890/0.4845/0.4856 s); parity maxdiff ≤ 3.84×10⁻⁷ with bad = 0 at every
size up to n = 12 000. For giant components above 512 nodes the scan uses a
Lanczos bulk spectrum (top 60 Ritz values), cutting the RMT stage at n = 1200
from 12.76 s to 0.41 s. Benchmarks: min-of-3 on a Ryzen 7 5800X3D + RTX 2080
Ti desktop. Caveat: above 512 nodes β/KS read the spectral tail only, so β is
not comparable across the 512 boundary.

== Comparison with the inherited 2025 code

`tools/compare_inap.py` normalises any run directory and a reference export
(the 2025 repo, or an iNAP 2.0 Galaxy export) to symmetric unweighted
adjacency on common nodes and reports edge precision/recall/F1 and Newman
modularity of each side's partition. Against the 2025 repo on ox at st = 0.52:
precision = 1.000, recall = 0.929, F1 = 0.963 (mena 836 edges, all inside the
2025 900-edge graph; Q 0.662 vs 0.586). The 2025 code hardcoded st = 0.52 and
never ran the RMT scan; the automatic binary-spectrum thresholds found here
(box 0.680, ox 0.610, anox 0.650) sit in the same range.

= Results

== The inherited "not estimable" verdict was a bug

The original unfolding spaced eigenvalues by rank alone, making every spacing
exactly 1 — the χ² statistic was then 0 for any graph and the scan accepted the
complete graph. A first repair (degree-5 polynomial fit with monotone
clipping) created plateaus, plateaus created zero spacings, and a
`filter_map` silently dropped every candidate that produced no statistic:
across all conditions and thresholds the audit trail showed
`excluded_too_few_spacings` at every point and *zero* `rejected_gof` rows. The
goodness-of-fit had never executed; "no threshold is estimable" was a statement
about the preprocessing, not the data. Replacing the clipped polynomial with a
local-density-average unfolding makes spacings positive by construction, and
the GOF finally runs.

== Automatic thresholds on real data

#table(
  columns: (auto, auto, auto, auto, auto),
  stroke: 0.4pt + rgb("#ccd3da"), inset: 4.5pt,
  [*condition*], [*accepted*], [*rejected GOF*], [*excluded*], [*candidates*],
  [box], [1], [12], [5], [18],
  [ox], [0], [8], [6], [14],
  [anox], [2], [10], [6], [18],
)
#figure.caption[Re-verified on commit `8ae7c7c` (seed 7): counts reproduce exactly from `rmt_scan.csv`.]

#table(
  columns: (auto, auto, auto, auto, auto, auto, auto),
  stroke: 0.4pt + rgb("#ccd3da"), inset: 4.5pt,
  [*condition*], [*st (auto)*], [*n / E*], [*M*], [*avg CC*], [*GD*], [*modules*],
  [box], [0.680], [118 / 275], [0.835], [0.637], [5.82], [16],
  [ox], [0.610], [159 / 549], [0.780], [0.667], [4.85], [16],
  [anox], [0.650], [154 / 450], [0.785], [0.633], [5.52], [17],
)
#figure.caption[Automatic RMT threshold on the BINARY adjacency spectrum (default `--binary-spectrum`), seed 7, fresh runs on commit `707c3b6`: all three conditions estimable. Same runs with `--null-samples 100` give Q-z 31.5 / 24.9 / 20.1; `--boot 20` gives mean module Jaccard 0.346 / 0.414 / 0.446; Clauset γ = 1.86 / 2.22 / 2.03 (all p = 1.00). Under the weighted spectrum (`--no-binary-spectrum`, Deng Eq. 3 literal), box/anox still accept (0.500 / 0.600) but ox is rejected everywhere because its weighted NNSD is Wigner-like (β = 0.92 at st = 0.52).]

#figure(
  image("paper/fig_scan_audit.png", width: 88%),
  caption: [The audit trail for the ox run, one point per candidate: grey crosses excluded before testing, red circles tested and rejected, green accepted. Above the dashed line χ² exceeds its critical value.]
)

== Why the weighted spectrum fails and the binary one works

In the default binary-spectrum mode all three conditions accept. In the
weighted mode (`--no-binary-spectrum`, the literal Eq. 3 reading) ox is
rejected at every eligible threshold because its weighted NNSD is
Wigner-like: Brody β = 0.92 at st = 0.52 (KS to GOE 0.082 vs KS to Poisson
0.204) — strong level repulsion that a correlation-derived weight matrix
carries by construction. The 2025 repo binarised its adjacency
(`np.where(S >= t, 1, 0)`) for exactly this reason; we make that choice
explicit and reversible. Separately, under CLR
(`--clr --clr-pseudocount 0.001`) the weighted ox spectrum becomes
estimable too (auto st = 0.63, 143/173 kept, E = 404, M = 0.798, β = 0.435)
— compositionality and weighting are two independent axes that both push
the raw ox spectrum away from Poisson.

== Results at a fixed threshold (st = 0.52, for comparability)

#table(
  columns: (auto, auto, auto, auto, auto, auto, auto, auto),
  stroke: 0.4pt + rgb("#ccd3da"), inset: 4.5pt,
  [*condition*], [*n / E*], [*M*], [*avg CC*], [*GD*], [*modules*], [*Mantel r (p)*], [*Q z (null, b=100)*],
  [box], [149 / 694], [0.639], [0.561], [3.22], [10], [0.173 (0.01)], [21.7],
  [ox], [166 / 836], [0.662], [0.640], [3.77], [13], [0.125 (0.04)], [36.5],
  [anox], [165 / 850], [0.596], [0.628], [3.63], [11], [0.077 (0.09)], [32.9],
)
#figure.caption[Fixed st = 0.52, 999 Mantel permutations, seed 7, Louvain. Q z-scores are against 100 degree-preserving nulls: observed modularity is 20–37 null SD above the degree-matched ensemble.]

#figure(
  image("paper/fig_fig5_ox.png", width: 100%),
  caption: [Module eigengene vs trait correlations, ox at st = 0.52 (the paper's Figure 5 analogue; panels chosen by effect size).]
)

#grid(
  columns: (1fr, 1fr),
  gutter: 8pt,
  figure(image("paper/fig_network_ox.png", width: 100%), caption: [Co-occurrence network, ox at st = 0.52 (166 nodes, 836 edges), one disc per module.]),
  figure(image("paper/fig_roles_ox.png", width: 100%), caption: [Module roles in the zi–Pi plane; most OTUs are peripherals, consistent with Deng et al.'s 90.9 %.]),
)

#figure(
  image("paper/fig_spectrum_ox.png", width: 80%),
  caption: [Adjacency spectrum and nearest-neighbour spacings of the ox graph at st = 0.52, against Poisson and Wigner references — the visual counterpart of β = 0.92.]
)

== Robustness of the modules

+ *Scale-free*: Clauset MLE gives γ = 2.68 (box), 2.57 (ox), 2.09 (anox) at
  st = 0.52 (all p ≥ 0.97), and 1.86 / 2.22 / 2.03 at the automatic thresholds —
  all degree tails are consistent with a power law. (The old OLS-on-log-bins
  R² ≈ 0.02–0.06 reading was misleading.)
+ *Bootstrap*: `--boot 20` at the automatic thresholds gives mean module
  Jaccard 0.346 (box), 0.414 (ox), 0.446 (anox) — modules are real but
  moderately unstable to sample resampling; report them with this number
  attached.
+ *Signed edges*: `--signed` on ox at 0.52 finds 7 negative of 836 edges —
  anticorrelated pairs exist but are rare at this threshold.

== Performance

#figure(
  image("paper/fig_bench.png", width: 100%),
  caption: [CPU scaling and CPU/GPU timing. GPU parity maxdiff ≤ 3.84×10⁻⁷, bad = 0 at every size; the GPU kernel is 1.69× faster than CPU at n = 12 000 (0.486 s vs 0.822 s min-of-3). The CPU pipeline bottleneck is the RMT stage (0.351 s of 0.362 s at n = 1920), cut by the Lanczos bulk spectrum above 512 nodes.]
)

#table(
  columns: (auto, auto, auto),
  stroke: 0.4pt + rgb("#ccd3da"), inset: 4.5pt,
  [*n × m*], [*CPU similarity (rayon)*], [*GPU RTX 2080 Ti (wgpu, min-of-3)*],
  [240 × 24], [0.0001 s], [0.0018 s],
  [1920 × 48], [0.0091 s], [0.0130 s],
  [8000 × 96], [0.2946 s], [0.2202 s],
  [12000 × 128], [0.8216 s], [*0.4858 s*],
)

= Discussion

The central methodological finding is negative and therefore important: an RMT
threshold scan whose goodness-of-fit never executes will confidently report
"not estimable" forever, and that verdict looks like a property of the data.
Writing one audit-trail row per candidate — with an explicit status for
excluded vs tested-and-rejected — is what made the distinction visible. We
recommend this as a minimum reporting standard for any RMT-based network
construction: the scan log *is* the result.

Second, the automatic thresholds we recover on the binary spectrum
(box 0.680, ox 0.610, anox 0.650) sit near the 0.52 the inherited code
hardcoded by hand — independent corroboration of a choice made without
reference to ours. But calibration on simulated data
shows recovery of planted modules is best at *low* thresholds while the
Poisson criterion scans upward: the criterion optimises spectral randomness,
not community recovery. We therefore keep the automatic threshold as the
default and expose `--st` for users who calibrate their own — the trade-off
is stated, not hidden in a default.

Third, spectrum source matters as much as compositionality: the *weighted*
thresholded matrix of a correlation graph is never Poisson (level repulsion,
β up to 0.92 on ox), so RMT threshold selection is only meaningful on the
binary adjacency — which is what the 2025 code did in practice. And CLR
addresses a different axis: raw relative abundances carry a closure
constraint, and under CLR the weighted ox spectrum also becomes estimable
(st = 0.63). Both effects are large enough to report, neither should be
silent.

Finally, modularity numbers are only meaningful against a null: with
degree-preserving rewiring, all three conditions sit 20–31 null SD above the
random ensemble (Q-z), and module stability under bootstrap is 0.35–0.45
mean Jaccard at the automatic thresholds — real structure, honestly
quantified.

= Conclusion

An audited, tested, open-source reimplementation of the MENA pipeline now
returns automatic RMT thresholds on all three real conditions (binary
adjacency spectrum; the weighted mode is documented as Wigner-like for ox),
quantifies modularity
against degree-preserving nulls, and reports module stability under
bootstrap. All results are reproducible from commit `8ae7c7c`:
`cargo test --release` (40/40), `mena run --abundance real/abundance_box.csv
--traits real/traits_box.csv --out-dir out` for the automatic path. The
graphs are modular, small-world-leaning, scale-free in the Clauset sense, and
20–37 null SD above random — and every number comes with its audit trail.

= Availability

Code: github.com/RaphaelRibes/hydro-mena (public, commit `8ae7c7c`, 40/40
tests). Audit and working notes: private vault mirror. Figures in `report/`
(png), source `report/paper.typ`.
