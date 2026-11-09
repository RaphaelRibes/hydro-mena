# Plan: reframe "failure/defect" language across hydro-mena docs

**Goal:** Keep the methodological finding (silent-GOF failure mode + scan-log-as-result)
but stop attributing bugs to the inherited 2025 code; attribute caught bugs to our own
round-1 build, and keep neutral comparison framing for the 2025 repo.

**Scope:** README.md, REVIEW.md, report/mena-audit.typ, report/paper.typ (repo);
vault pages listed below. No src/ changes, no number changes, no figure changes.
Tests must stay 41/41; both PDFs recompile; commits per file group, push to
origin HEAD:main (repo) and vault HEAD:main.

## Decision (locked by user 2026-11-09)
- "Defect" language about *our own first build* stays (it is the audit's evidence).
- "Broken/inherited/bug" language about the *2025 repo* goes. The 2025 repo is cited
  only as: (a) reference network for parity comparison (neutral), (b) evidence that
  binarising the scan spectrum is standard practice (cite `np.where(S>=t,1,0)` as
  "the reference implementation binarises", not "because theirs was broken").
- Audit §"Pipeline and where it failed" retitled to "Pipeline and where the first
  build failed"; "before column" captions say "our round-1 build" not "broken".
- Paper §"The inherited 'not estimable' verdict was a bug" retitled to
  "Why 'not estimable' was an artefact of the first build" — the verdict was ours
  (round 1), not the 2025 repo's; the 2025 repo never claimed RMT on this data,
  it used a fixed 0.52.

## Task 1: audit typ — reframe sections (report/mena-audit.typ, ~13 hits)
- L29: "what was broken, what was fixed" -> "what the first build got wrong, what the fixed estimator now says"
- L39: "audit of an inherited implementation" -> "audit of our first build (round 1), which itself inherited reported results from the 2025 code"
- L56: "The tool was inherited with reported results" -> "Reported results existed from the 2025 code; our round-1 build reproduced them, and the audit's job was to check the estimator itself"
- §"Pipeline and where it failed" (L78) -> "Pipeline and where the first build failed"; L91/L94 diagram labels "two defects" -> "two round-1 defects"
- §"Defects found" (L98) -> "What the first build got wrong"; L112 "second, quieter defect" -> "second, quieter failure of the same stage"
- L135 caption "under the broken rank" -> "under the round-1 rank-only unfolding"
- §"Other defects" (L148) -> "Other round-1 issues"; table header "defect" -> "issue"
- L397: "2025 repo's scan binarised" -> "the reference implementation binarises the scan spectrum (`np.where(S >= t, 1, 0)`)" — neutral
- L467: same neutral reframe for `--binary-spectrum`
- L493: "run against ... the 2025 reference repo. Against the 2025 repo on" -> keep (neutral comparison), maybe "reference implementation"
- Verify: `grep -c -i -E "defect|broken|papered|inherited" report/mena-audit.typ` drops from 13 to <= 4 (remaining ones must be "our round-1" attributions)
- `cd report && typst compile mena-audit.typ mena-audit.pdf`; pdfinfo Pages == 9; pdftotext grep for "round-1" present, "inherited.*broken" absent
- Commit: `docs(audit): reframe failure language as round-1 findings, 2025 repo cited neutrally`

## Task 2: paper typ — reframe (report/paper.typ, ~8 hits)
- L35: "audited the inherited 2025 code before" -> "audited our round-1 build (which carried the 2025 reported results) before"
- L39/L68-71: "the inherited scan's χ² never ran, so its 'not estimable' ... was broken" -> "our round-1 scan's χ² never executed, so 'not estimable' was a bookkeeping artefact, not a data property"
- §"Comparison with the inherited 2025 code" (L160) -> "Comparison with the reference implementation"; keep numbers, neutral wording
- L167: "The 2025 code hardcoded st = 0.52" -> "The reference implementation used a fixed st = 0.52"
- §"The inherited 'not estimable' verdict was a bug" (L173) -> "'Not estimable' as an artefact of the first build"; body: the verdict came from our round-1 scan, not from the reference implementation (which never ran RMT on this data)
- L103: "matching the 2025 repo's fig_5_threshold.py" -> "matching the reference implementation's scan"
- L221: "The 2025 repo binarised its adjacency" -> "The reference implementation binarises the scan adjacency"
- L272: "the 0.52 the 2025 code hardcoded" -> "the 0.52 used by the reference implementation"
- L316/L332: "the inherited code hardcoded" / "what the 2025 code did" -> "the reference implementation's fixed 0.52" / "what the reference implementation does"
- Verify: `grep -c -i -E "defect|broken|inherited|2025 code" report/paper.typ` from 8 to <= 2
- `typst compile paper.typ paper.pdf`; Pages == 7; pdftotext checks
- Commit: `docs(paper): neutral reference-implementation framing; round-1 attribution for caught bugs`

## Task 3: README.md + REVIEW.md
- README.md currently 0 hits — verify and leave unchanged unless a new hit appears; commit only if changed
- REVIEW.md L13: quote is fine if it describes our conclusion, but reword to "the round-1 'RMT stage broken' conclusion"; L52: "This is why the 2025 repo's author hardcoded 0.52" -> "This is why the reference implementation uses a fixed 0.52"; L79: "Against the 2025 repo's reference network" -> "Against the reference network"; §3 "Other defects found" -> "Other issues found (round 1)"
- Commit: `docs: neutral REVIEW.md wording`

## Task 4: vault pages
- `01-Project/rmt-threshold.md`: keep historical facts but reframe L6: "the 2025 repo's own fig_5_threshold.py uses a different..." -> "the reference implementation uses a histogram-based χ²..."; "the 2025 author abandoned RMT" -> "the reference implementation abandoned RMT for a fixed st = 0.52"
- `05-Benchmarks/cpu-scaling.md` L2: "produced by the broken RMT" -> "produced by the round-1 RMT failure"
- `06-Reference/rmt-method-decision.md` L46: "correct, not a bug to 'fix'" -> keep the conclusion, reword "not a bug" -> "consistent with the reference implementation"
- `07-Code-Changes/code-changes-2026-11-08*.md`: scan for "defect|broken|bug|inherited" — reframe any that blame the 2025 repo; ours stay
- Commit per file or one vault commit: `docs: neutral reference-implementation framing across vault`
- Push vault HEAD:main

## Task 5: final verification
- `cargo test --release` -> 41 passed (no src changes, but confirm)
- Both PDFs recompiled, page counts unchanged (9 / 7)
- Repo: `git push origin HEAD:main`; vault: `git push origin HEAD:main`
- gh api verify both remote shas == local HEAD
- Final grep across repo docs + vault: no remaining "the 2025 (repo|code).*(broken|bug|defect)" or "inherited.*broken" phrasing
