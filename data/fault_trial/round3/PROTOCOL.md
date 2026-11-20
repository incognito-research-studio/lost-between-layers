# Fault-injection evaluation with separated roles: round 3 protocol

Written by Claude (the architecture role), 13 November 2026, after the seventh external review
and after rounds 1 and 2 were unblinded, and before any round-3 fault, episode or negative case
existed. The protocols of rounds 1 and 2 (`data/fault_trial/PROTOCOL.md`,
`data/fault_trial/round2/PROTOCOL.md`) apply except where this file says otherwise. Rounds 1
and 2 stay reported as they are.

## Why a third round

Round 2 showed checks missing for obligations the architecture states: no check compared the
intervals a decision used with those the store holds, the restart check compared the number of
reloaded deltas rather than their content, and unit, calendar, authority, version and
commit-time precedence faults passed unnoticed. The review asked for repaired checks reported as
a separate configuration, with the known faults as regression evidence and fresh, withheld
faults as evidence of generalization, and for actual containment to be measured rather than
inferred from a check's timing tag.

## Instruments (frozen before the round-3 generator starts)

| | Rounds 1 and 2 | Round 3 |
|---|---|---|
| Checks | `ft_harness.py` (solver < generic < full) | the same, unchanged, plus `tools/ft_repaired.py` (configuration `repaired`, eleven checks) |
| Runs | monitor only | monitor and gating: with gating, the repaired configuration's absolute pre-execution checks hold a release that fails them through the pipeline's L5 monitor hook; the gate fails closed |
| Driver and scoring | `ft_round2.py`, `ft_trial.compare`, `ft_score.py` | `tools/ft_round3.py` (the same steps; `compare` over four configurations, plus harm and containment) |
| Effect trace | extended (`ft_trace_ext.py`) | the same |
| Worlds | e1v2 replay, 12 scenario episodes, the generator's episodes | the same fixed worlds, plus round 3's episodes |
| Generator | a fresh agent, runnable sandbox (`ft_sandbox2.py`) | a different fresh agent, the same sandbox with the round-3 brief (`tools/ft_sandbox3.py`) |
| Negative cases | six, three kinds | twelve, six kinds (three new boundary kinds) |

The repaired checks were written after rounds 1 and 2 were unblinded, and were extended and
adjusted on those rounds before this freeze: two checks were added after their first
regression runs (`commit_precedence` for round 2's faults 06, 17 and 24, `recheck_complete` for
round 1's fault 14), `interval_store` was changed to select facts itself rather than through
the store's own lookup (round 1's fault 02), the unit comparison normalizes spellings (the
clean e1v2 replies write "units per week"), the robust check uses the margin a gate logs, the
declared policy reaches the run before its first event, and the gate fails closed. Their results on
rounds 1 and 2 (`results/fault_trial_r1_repaired.json`, `results/fault_trial_r2_repaired.json`)
are therefore regression evidence only. After the freeze nothing in the instruments changes;
a defect found later is recorded as a deviation and reported with the frozen results.

One check considered and left out: comparing each record's C-id with a declared, reviewed
encoding would flag every change to a formulation's code, because the C-id hashes that code;
it would detect the patch, not the failure, so it is not part of `repaired`.

## Withheld from the round-3 generator

As in round 2 (`src/dsoi/properties.py`, `src/dsoi/trace.py`, the real
`src/dsoi/layers/l5_monitors.py`, everything under `tools/`, `tests/` and `results/`, the
paper, the review files, the protocols), and in addition the fault sets of rounds 1 and 2
(`data/fault_trial/set-1/`, `data/fault_trial/round2/set-2/`), their manifests and audits,
the adjudication files and the repaired checks. The generator gets only the sandbox. Its tool
calls are audited afterwards as in rounds 1 and 2.

The sandbox is built by `tools/ft_sandbox3.py`: round 2's builder, unchanged, with the round-3
brief. The round-1 builder it calls copies the layer contracts from `mdpi-paper/tables/`, which
moved to `manuscripts/mdpi-paper/tables/` on 12 November 2026; `ft_sandbox3.py` redirects that
one path after checking that the moved file is byte-identical to the one rounds 1 and 2 used.

## Outcomes (unit: the fault)

Primary, per configuration over round 3's effective faults: faults detected; faults detected by
a pre-execution check; harmful runs (more violations of the true constraints than the clean run)
flagged. The paired comparison of `repaired` with `full` is reported as counts and an exact
McNemar test, which describes this fault set only; `repaired` contains `full`, so it cannot
lose a fault `full` caught.

Containment, with the gate on: harmful runs whose violations do not exceed the clean gated
run's; harmful faults with every harmful run contained; holds on patched runs. Cost: holds on
the clean worlds (releases held that would not have violated the true constraints count as lost
availability) and on the negative cases, and false alarms of each configuration on the negative
cases. Twelve negative cases cannot establish a false-alarm rate; zero observed is reported as
zero observed.

Per-run detail (which checks fired on which run, violations without and with the gate, holds)
is kept for every effective fault, and after unblinding a reader compares each observed effect
with the manifest, as in round 2's adjudication. Results on the fixed worlds and on the
generator's episodes are reported separately.

Episode screening, the determinism check on the clean pipeline, the ALLOWED rule, the exclusion
of unbuildable episodes and the classification of a patch as effective or inert are as in round
2.
