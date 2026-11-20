# Fault-injection evaluation with separated roles: round 2 protocol

Written by Claude (the architecture role), 8 November 2026, after round 1 (set 1, results
committed in 447f857) and before any round-2 fault, episode or negative case existed. Round
1's protocol (`data/fault_trial/PROTOCOL.md`) applies except where this file says otherwise.
Round 1 stays reported as it is.

## Why a second round

In round 1, eight of 26 patches never changed behaviour because the frozen worlds never
exercised their code paths (all three stale-approval patches among them), and the effect
check could not see replay or reload outputs (round 1, deviation 1). Round 2 addresses both
prospectively: the generator writes the input episodes that exercise its faults, and the
extended effect trace is part of the protocol from the start.

## What changes from round 1

| | Round 1 | Round 2 |
|---|---|---|
| Checks, configurations, tags, classification, scoring | `ft_harness.py`, `ft_trial.compare`, `ft_score.py` | the same files, unchanged (byte-identical to `data/fault_trial/FREEZE.json`) |
| Effect trace | store, records, ingests, approvals | plus replay outputs and the reloaded log (`tools/ft_trace_ext.py`), adopted before round 2 |
| Worlds | e1v2 replay (672 runs), 12 fixed scenario episodes | the same, plus the generator's episodes |
| Generator sandbox | source to read, not runnable | runnable: an interface-only stub for `l5_monitors.py`, a behaviour-only runner (`run_case.py`), a venv with the pinned numpy, scipy and pyyaml (`tools/ft_sandbox2.py`) |
| Generator's task | patches, negatives, manifest | patches, episodes that exercise them, negatives, manifest |
| Generator | fresh agent | a different fresh agent, with no access to round 1's fault set |

Withheld from the round-2 generator: `src/dsoi/properties.py`, `src/dsoi/trace.py`, the real
`src/dsoi/layers/l5_monitors.py`, everything under `tools/`, `tests/` and `results/`, the
paper's checkability table and Section 10.3, the review files, both protocols, and
`data/fault_trial/set-1/`. Its tool calls are audited afterwards as in round 1.

The checks were not changed after round 1, although round 1 showed two gaps (no check that
the gate receives the intervals the store holds; the restart check compares the number of
reloaded deltas, not their content). Round 2 tests the same instrument on new faults and new
worlds; a revised check set would be a third, separately reported configuration.

## Episode screening (fixed)

Before any patch runs, every episode is run on the clean pipeline. An episode whose pipeline
cannot be built there (a malformed specification or fixture) is excluded and listed; nothing
else about an episode is screened, and an episode on which a check fires on the clean
pipeline stays in (only new firings count). Episodes, like the fixed worlds, are run on the
clean and on every patched pipeline. The manifest's statement of which episodes exercise
which patch is not read by the trial.

## Results

Primary for round 2: `results/fault_trial_r2.json` and its scores from `tools/ft_score.py`,
unit = the fault, as in round 1. Across rounds, results are reported per round; a pooled
count over both rounds' effective faults is descriptive only. After unblinding, each
effective patch's observed change is compared with the manifest's intended effect by a
reader, and the episodes the manifest lists for each inert patch are named.
