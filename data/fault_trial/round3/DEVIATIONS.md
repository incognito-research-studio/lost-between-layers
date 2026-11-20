# Round 3: deviations from the frozen protocol

## 1. `robust_release` did not cover the state-vertex gate (found after unblinding, 13 Nov 2026)

`tools/ft_repaired.py` lists the rungs whose releases `robust_release` checks as
`ROBUST_RUNGS = ("box_robust", "vertex")`. The state-vertex gate's rung is `"state_vertex"`
(`src/dsoi/layers/l5_gate.py`), so releases of that gate were never checked, in rounds 1-2's
regression runs (fixed scenario S8) and in round 3. Found when the adjudication looked at
fault 05, which breaks the state-vertex gate and was missed.

The frozen instrument is not changed and its round-3 results (`results/fault_trial_r3.json`)
stay primary. A post-hoc analysis with the rung name corrected and nothing else changed
(`tools/ft_r3_deviation1.py`, `results/fault_trial_r3_dev1.json`) is reported separately and
labelled as such; it reruns every patch, the clean worlds and the negative cases, so any
false alarm the correction adds is visible.

## 2. `l1_reference` counted its own crash as a firing (found by an independent review, 13 Nov 2026)

`c_l1_reference` in `tools/ft_repaired.py` sorts `(key, fact)` pairs. When two committed facts
share a key, Python compares the facts themselves and raises `TypeError`; `_fires` counts any
exception as firing, in the monitor and in the gate. In round 3 this happened only on fault 14
(two monitor runs, four gated runs). The fault is also detected by `commit_precedence`.

The frozen results stay primary. A post-hoc rerun with the sort on the key alone, nothing else
changed, and every exception a check raises logged (`tools/ft_repaired_dev2.py`,
`tools/ft_r3_deviation2.py`, `results/fault_trial_r3_dev2.json`,
`results/fault_trial_r3_dev2_exceptions.json`) is reported separately. Outcome: every round-3 result is byte-identical to the frozen one
(all 26 fault entries including per-run detail, the clean runs and the 12 negative cases). On
fault 14 the corrected check fires on the same two runs by its own comparison, so the crash did
not create a detection. The exception log shows that the only other checks that raise are
`restart` and `replay_store` (07, 13, 15) and `restart_content` (13), where the patched code
raises on reload or replay.

## Findings after unblinding that are not deviations (independent review, 13 Nov 2026)

An agent given the review, the instruments, the fault sets and the results, but not the
architecture role's conclusions, reviewed round 3. Besides deviation 2:

- **Overlap with rounds 1 and 2.** 8 of the 26 round-3 patches are identical to a patch of
  round 1 or 2 (same removed and added lines after stripping whitespace and comments), and 9 more
  change the same source line; round 2 already repeated round 1 (5 identical, 6 same line, of
  25). Withholding the earlier fault sets did not make the faults new. `tools/ft_overlap.py`
  computes the classes and the results by class (`results/fault_trial_overlap.json`): on the 9
  faults with no earlier counterpart, full detects 6 and repaired 7 (McNemar p = 1.0; with
  deviation 1 corrected, 8 and p = 0.5). The protocol did not foresee this; the overlap is
  reported with the results, and the round-3 totals are no longer described as evidence that
  the repaired checks generalize.
- **The clean hold on `episodes/alias-model-update` is a false hold.** The clean pipeline
  refuses the answer of an undeclared model version, as it should; the frozen `version` check
  fires, and the gate holds a release that would not have violated the true constraints. It is
  reported as a false hold on one of the 21 fresh clean episodes, not only as lost availability.
- **Negative-case coverage.** No negative case declares a span or block conflict policy, a
  recheck or the state-vertex gate, so `span_cover`, `blocked_read` and `recheck_complete` were
  not exercised on them.
- **Source hash method.** `FREEZE.json`'s `src_dsoi_sha256` is
  `tools/ft_round2.py: dir_sha(src/dsoi)`: SHA-256 over the concatenated hex SHA-256 of every
  file under `src/dsoi`, in sorted path order. It reproduces on the committed tree.
- **Runner selection.** `run_case.py` (round 2's, reused) shows the generator replay outputs
  and whether the reloaded log's deltas match, and the brief asks that every patch change some
  runner output; this favours faults those outputs reveal.
