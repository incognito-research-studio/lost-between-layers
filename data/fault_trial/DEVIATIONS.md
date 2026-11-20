# Fault trial, set 1: deviations from the frozen protocol

Recorded by the architecture role (see `PROTOCOL.md`), 8 November 2026. The frozen result,
`results/fault_trial_v2.json` and `results/fault_trial_v2_scores.json`, is the primary
result. Nothing below edits a frozen instrument (`data/fault_trial/FREEZE.json`); the SHA-256
of every frozen file was checked before the reanalysis and matched.

## Order of events

1. Instruments frozen (commit ce3bb13, `FREEZE.json`).
2. Sandbox built (`SANDBOX_COMMIT`); the fault-generation agent wrote 26 patches, 6
   negative cases and `manifest.json` (SHA-256 2645c626...cc4d) and reported only those
   counts and the hash.
3. Its tool calls were audited (`AUDIT.json`): 156 calls, none reading, searching or writing
   outside the sandbox.
4. `tools/ft_trial.py` ran with the manifest unopened; then `tools/ft_score.py` opened it.
5. After unblinding, the deviation below was decided and run.

## Deviation 1: extended effect trace (decided after unblinding)

**Problem.** The frozen effect check digests the store, the decision records, the ingests and
the approvals after each run. It does not see what the system returns when it replays a
decision or reloads its saved audit log. A fault whose only effect is on replay or on the
reloaded log is therefore classified inert and excluded before detection is looked at,
although replay and restart are two of the seven fault families the protocol named. This was
noticed only after the frozen results were unblinded: 13 of 26 patches were inert, including
every `audit_restart` and `replay_observations` patch.

**Change.** `tools/ft_trace_ext.py` imports the frozen harness unchanged and replaces only the
effect trace with one that also covers, for every decided record, the action
`Pipeline.replay` returns from the logged state and from the store, and the content of the
log after `to_jsonl` and `from_jsonl`. These are outputs of the system's own functions, not
of a check. `tools/ft_reanalysis.py` reruns every patch with the frozen `ft_trial.py`
functions and this trace; `ft_score.py` scores it against the same sealed manifest. The
checks, configurations, classification and scoring are unchanged.

**Effect.** Effective patches: 13 → 17 (four reclassified: one `replay_observations`, three
`audit_restart`). Full configuration: 7 of 13 → 10 of 17 detected; generic: 1 of 13 → 1 of 17;
solver: 0 → 0. Exact McNemar, full against generic: p = 0.031 → 0.0039. False alarms on the six
negative cases: none, before and after.

**Risk of bias.** The decision was made knowing which patches were inert and that checks had
fired on some of them; it can only add effective patches, and it adds the ones in the two
families that the frozen trace could not see. Reported as secondary.

**Descriptive addition.** `tools/ft_inert_firings.py` lists checks that newly fire on the nine
patches still inert under the extended trace (`results/fault_trial_v2_inert_firings.json`):
one (an evidence-only fault, the observations dropped from the logged tuple) is flagged by C2
on the S1 recheck episode although no output of the system changes; the other eight trigger
no check. Not counted as detections.

## Known gaps, not deviations

- **World coverage.** Eight patches stay inert because the frozen worlds never exercise their
  code paths: all three `stale_approval` patches (no expired, reused or wrong-variable approval
  occurs in the S2 episode), a version comparison (no second version of the same model is
  served), the supersession ranking (no master correction occurs outside the negatives, which
  run on clean code only), the vertex gate at an interval's top, recheck replay for lot and
  warehouse models, and unauthenticated approvals on save. A second round in which the
  generator also writes the input episodes that exercise its faults, frozen before any patch
  is run, would address this prospectively.
- **Intended-effect check** (protocol, scoring): comparing each effective patch's observed
  change with the manifest's intended effect is pending a reader.
- **Realism rating** of a sample of patches by a coauthor who did not write the checks:
  pending.
