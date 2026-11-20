# Blind fault trial: protocol

Design of 4 November 2026, superseded by `data/fault_trial/PROTOCOL.md`. Nothing here has
been run with real faults.

## Question

The paper shows two log predicates that find the two harness defects of the replay, but
they were written after those defects were known. This trial asks whether the checks the
architecture already defines detect integration faults that the people who wrote the
checks did not know about.

## Roles

- **Planter.** One person who did not write `tools/e1v2_monitors.py` or the log
  predicates. A co-author is acceptable if they have not read the monitor code since
  4 November 2026. Their name and role go in the results.
- **Runner.** Anyone. The runner receives only the patch files, never the key.

## What is frozen before any fault is written

- `tools/e1v2_monitors.py` (the monitor set, `MONITORS`), with its SHA-256 recorded in the
  results by `tools/fault_trial.py`.
- `tools/fault_trial.py` and the classification it applies: rejected, crashed, inert,
  detected, missed.
- The stored replies, cases and margins it runs on.

A change to any of these after the faults exist is a deviation and is reported as such.

## What the planter writes

1. Between 8 and 12 faults, each one file `NN-anything.patch`, a unified diff against
   `src/dsoi` with paths relative to `replication-package/` (`--- a/src/dsoi/...`,
   `+++ b/src/dsoi/...`). Patches may not touch `tools/`, `tests/`, `configs/`,
   `scenarios/`, `data/` or `results/`. File names must not describe the fault.
2. Each fault should be one an implementer could plausibly make when wiring layers
   together: wrong source ranking, a lost or reordered message, a unit or calendar
   conversion, an off-by-one in valid time, a gate reading the wrong bound, a record
   written before execution, a stale cache, a dropped provenance field, and so on. Faults
   inside one layer's arithmetic count too. Faults written to defeat a particular monitor,
   or patches that only edit the monitors' inputs, do not.
3. A sealed key, `KEY.md`, giving each patch's intent in one or two sentences. Before
   handing over the patches, the planter sends the SHA-256 of `KEY.md` to the runner (or
   records it with a timestamp outside the authors' files). The key is opened only after
   `results/fault_trial.json` is written.

## Running

```
uv run python tools/fault_trial.py --faults path/to/faults --out results/fault_trial.json
```

The tool copies `src/` for each patch, applies the patch there and runs the monitors in a
fresh interpreter. It checks that the patched package is the one imported, and stops if the
unmodified pipeline crashes. It takes a few seconds per fault on one core.

## Reporting

- Counts: detected, missed, inert, crashed, rejected; detection rate =
  detected / (detected + missed). Report the inert and crashed faults; do not drop them.
- For each fault, after opening the key: its intent, which monitors changed, and which
  outcome counts changed.
- A fault that changes outcomes only through the benchmark truth (violations) and no
  monitor is a miss. These are the informative cases: they show which faults the log
  cannot reveal.
- Limits: one harness, one world, stored replies; faults chosen by one person; a high
  detection rate does not show that operational faults would be found.

## Demonstration only

`tests/fault_demo/` holds two demonstration patches: the known source-priority defect
(detected by `shadowed` and `committed_unread`) and a docstring change (inert). They test
the harness and are not evidence for the trial.
