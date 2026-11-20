# E1 v2: deviations from the frozen protocol

This file is not part of the freeze; it records every change to a frozen file made after
the replies were collected, with the reason. The summary (`results/e1v2/summary.json`)
lists the changed files under `deviations_since_freeze`.

## Correction status (3 Nov 2026)

Historical results and the original manifest are retained. The principal reported result
now comes from `tools/e1v2_corrected.py`, stored separately in `results/e1v2_corrected`.
It is explicitly exploratory and contains no new model calls or fresh held-out cases.
It combines both corrections: single-source notices outrank the baseline master, and
widening occurs as a versioned gate transformation after interpretation, so no changed
interpreter label can reject a widened reading and no changed state reaches the model's
second message. Every original input snapshot is checked; all match. Margins are refitted
from corrected calibration readings. That selection rule remains descriptive and does
not establish a conformal guarantee. Emitted-value diagnostics, resolved-state coverage
and executed outcomes are reported separately, with approximate case-clustered intervals.

Further changes under `src/dsoi` repair C2 and replay for execution-time observations,
persist all audit evidence, and capture immutable model/runtime metadata in future local
collections. These changes postdate the freeze and are not retroactive evidence about the
historical runtime. No old weight digests, quantization, hardware details or timestamps
have been invented. The historical scorer remains intentionally available to reproduce
the historical defective arms. New regression tests cover assertions that formerly were
shadowed, positive-margin widening without a nu_I mismatch, mismatched input snapshots,
changing rechecks during replay, audit save/load and altered or incomplete evidence.

## 1. Scorer: cycles that end in an error (23 Oct 2026, after all local replies)

**File:** `src/dsoi/experiments/e1v2_score.py` (`_run_arm`, `arm_row`, `summarize`).

**What happened.** Scoring the three local models stopped with an exception. In 14 of
phi4's case-repeats (3 arms each), the reply asserts the new capacity and then retracts
the old value for the same weeks. Retraction works by key, not by value (Def. 2; the value
in a retract op is ignored), so the delta leaves those weeks without capacity. L4 cannot
build the plan (`MissingState`), L6 logs the cycle with outcome "error", nothing executes,
and the pipeline re-raises, as its contract states (`Pipeline.decide`). The frozen scorer
did not catch the exception, so no model could be scored.

**Change.** `_run_arm` catches an exception from the decision cycle only if L6 logged
that cycle as "error" (anything else is re-raised) and scores the logged record: no
execution, so not released, no violation and no cost. Each arm's row gains
`error_<arm>`, and the summary reports its rate per arm. The same deleted weeks then
stopped the interpretation measures (a week without a value had no interval), so a week
without a value now counts as changed and as not covered, and its calibration score is
the nominal capacity, the margin that spans the domain (`score_case`, `_covered`).
This choice sets phi4's margin: 6 of its 22 calibration scores are such weeks, so
q = 1000; without them q would be 50. Run with q = 50, the post hoc calibrated arm (item 2)
gives phi4 the same coverage (17 of 176) and violations (152 of 176); only the mean cost of
feasible releases changes (178 instead of 2284). Nothing else changed: no cases, prompts,
replies, thresholds or other measures. qwen2.5
and llama3.1 never reach these paths; every number in their summaries is identical to a
scoring before the change, and their rows differ only by the new `error_<arm>` columns,
all false.

**Why this is not a choice made after seeing results.** The treatment is the one the
pipeline already applies (an error cycle executes nothing); the alternative, dropping the
14 case-repeats, would contradict "No reply is excluded" in the protocol. The test
`tests/experiments/test_e1v2.py::test_a_delta_that_retracts_what_l4_reads_is_scored_as_an_error_cycle`
reproduces phi4's pattern.

## 2. Known defect in the pre-registered calibrated arm (not changed)

**What it is.** `widen` labels a widened reading with the interpreter version plus
`+cal<q>`, but `_run_arm` pins the pipeline's nu_I to the unwidened version. L1's contract
rejects a reading whose version differs from the pinned one, so for every q > 0 each
widened reply is logged as an abstention, and the calibrated arm decides on the nominal
state. The harness test used a mock whose margin was 0, where `widen` changes nothing, so
the defect was not caught before the freeze. Found on 23 Oct 2026 while scoring. q is
1000 for all three local models, so the arm was affected for every model: qwen2.5 has no
calibration case with a changed affected week and llama3.1 has 8 (the quantile needs 9),
so q takes the code's fallback; phi4 has 22, and its 21st smallest score is 1000.

**What is reported.** The pre-registered calibrated arm, as frozen. Beside it, labelled
post hoc, the arm the protocol describes, computed by `tools/e1v2_posthoc.py` (outside
`src/dsoi`, so the freeze is unaffected) with the same replies, the same q and the same
gate, the pipeline pinned to the widened label. The test
`test_frozen_calibrated_arm_rejects_widened_readings_and_the_post_hoc_arm_does_not`
shows both behaviours.

## 3. Post hoc measure: cost of feasible releases

The pre-registered cost ("mean shortfall above the true optimum when released") averages
over violating plans too. A plan that exceeds the true capacity produces more than the
plant can, so it looks cheaper than the true optimum, and the mean is negative when most
releases violate. The paper's table reports it as registered and adds, labelled post hoc,
the mean over released plans that are feasible for the true state
(`tools/e1v2_table.py`).

## 4. Post hoc arm: hold the decision after an abstention

Added on 24 Oct 2026 after the results showed that abstentions leave the state stale. The
arm is the box-robust gate on the stated intervals, with L2 holding the decision whenever
a message of the case ended in an abstention, as if the message waited in the approval
queue and the decisions of its class waited with it. It needs no new run: it is derived
from the ingest outcomes the stated arm logged (`results/e1v2/rows.csv`) by
`tools/e1v2_posthoc.py` (`hold_rows`), and a held case releases nothing. Its cost in
availability is reported as the share of quiet notices whose decision it holds. The test
`test_hold_on_abstention_releases_nothing_after_an_abstention` checks the rule.

## 5. Post hoc reanalysis: intervals clustered by case (3 Nov 2026)

The frozen scorer gives Wilson intervals over case-repeats, which treat the two repeats of a
case as independent. In a later reanalysis, `tools/e1v2_clustered.py` recomputes every
interval of the replay results with cases as clusters (Wilson on the case-repeats divided by
a design effect from the cluster-robust variance, bounded to [1, 2]), adds paired case-bootstrap
differences between arms and an exclusive classification of outcomes, and writes
`results/e1v2_clustered.json`. Rates are unchanged; `tools/e1v2_table.py` checks every k and n.

## 6. Harness defect found after scoring: asserts shadowed by the capacity master (not changed)

**Found** on 3 Nov 2026, while extracting the raw replies behind the outcome "no change",
after all results were known.

**What it is.** A single-source notice enters L3 from `email:e1`, which has no entry in the
frozen source priorities (`src/dsoi/experiments/e1v2_world.py`, `scenario`: `portal:supplier` 2,
`email:planner` 1), so it ranks
0, as does the authenticated capacity master `erp:capacity-master`. An interpreted `assert`
adds a fact beside the master's without closing it (Definition 3), and `StateLayer.resolve`
ranks by (source priority, authenticated, transaction time), so the master's nominal value
is read. The reading is committed but never used. Only `supersede`, or `retract` before
`assert`, replaces the master's value; the prompt (`src/dsoi/layers/l1_claude.py`) does not
say so, and the reference reader uses `supersede`. Conflict cases are unaffected, since their
sources rank 1 and 2.

**Evidence.** In the test part, 78 change-notice case-repeats were answered without
abstention, committed, and left plant X at nominal (llama3.1 35, phi4 39, qwen2.5 4). Every
one of them is a reply whose operations are all `assert`, on a single-source notice. No stored
reply, in either split, uses `supersede` (804 `assert`, 53 `retract`).

**What is reported.** The frozen results, unchanged. Beside them, post hoc,
`tools/e1v2_shadowing.py` re-scores the stored replies with one change made in memory:
`email:e1` ranks 1, above the master (`results/e1v2_shadowing.json`). Joint coverage of change
notices rises from 0, 4 and 9 of 176 case-repeats to 4, 12 and 42 (qwen2.5, llama3.1, phi4);
violations under the box-robust gate on stated intervals fall from 176, 171 and 160 to 172,
150 and 126; quiet notices stay 48 of 48 for every model, and every snapshot still matches.
Abstentions are unaffected (172, 139, 121).

**Fix.** Not applied to the frozen files. A new freeze must rank interpreted notices
consistently with the master, or state in the prompt that a change to a current value is a
`supersede`, and must test that a correct `assert`-style reading changes the state read.
