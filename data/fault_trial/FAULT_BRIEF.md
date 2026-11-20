# Brief for the fault-generation role

You write integration faults and legitimate-but-unusual cases for a reference implementation
of a layered decision system, so that others can test whether the system's own records
reveal faults they did not write. You work only in the sandbox you were given.

## The system

A planning pipeline in Python (`src/dsoi`). Messages from suppliers and planners pass
through an interpreter (L1), a procedural layer that admits changes and actions and checks
approvals (L2), a bitemporal state store with source precedence and conflict rules (L3), a
deterministic decision layer that builds and solves a linear program (L4), a release gate
that decides what may execute, optionally re-reading a system of record at execution time
(L5), and an audit log that records messages, dispositions, approvals, decisions and
execution reports and can be saved and reloaded (L6). A version vector pins the version of
everything that can change a decision. `CONTRACTS.tex` gives each layer's contract,
guarantees and residual failure modes. `scenarios/` holds worked examples with recorded
interpreter outputs (fixtures).

## Part 1: faults (about 25 patches)

Write faults an engineer could plausibly introduce while wiring or maintaining these layers.
Work from these failure descriptions, three or four faults per family, and up to three
others of your own:

1. **precedence_supersession.** The rules for which source wins are mis-wired: a lower-
   authority source wins, a newer value never displaces an older one, or a correction from
   the authoritative record is ignored.
2. **unit_calendar.** A value is converted with the wrong unit or calendar: per day against
   per week, working days, week boundaries, an off-by-one week index.
3. **dropped_week_entity.** An update loses part of its scope: a week at the edge of a range,
   one of two entities, or a key attached to the wrong entity.
4. **uncertainty_lost.** The interval attached to an interpreted value is narrowed, dropped
   or replaced by its point value on the way to the planner or the gate.
5. **replay_observations.** Replaying a past decision does not use the observations recorded
   when it was released: it reads live data, drops them, or reorders them.
6. **audit_restart.** After the audit log is saved and reloaded, some evidence is missing or
   altered: messages, dispositions, approvals, decision records, committed changes.
7. **stale_approval.** An approval or decision is applied after its validity has changed:
   an expired approval, an approval reused for a second action, an approval for another
   variable or role.
8. **other.** Any other plausible integration fault.

Rules:
- Each fault is one unified diff, `NN-<random word>.patch` (the name must not describe the
  fault), against the sandbox with paths `src/dsoi/...` (`--- a/src/dsoi/x.py`,
  `+++ b/src/dsoi/x.py`), applicable with `patch -p1` from the sandbox root. Make each from
  a clean tree (`git diff` in the sandbox, then `git checkout .`).
- Only these files may change: `canon.py`, `constraints.py`, `context.py`, `store.py`,
  `types.py`, `pipeline.py`, `registry.py`, `layers/l1_interpreter.py`, `layers/l1_claude.py`,
  `layers/l2_procedural.py`, `layers/l3_state.py`, `layers/l4_decision.py`,
  `layers/l5_gate.py`, `layers/l6_audit.py`, `experiments/e1v2_world.py`,
  `experiments/e1v2_corrected.py`.
- A fault must change behaviour on realistic inputs (the scenarios or capacity-notice
  cases), not only on inputs nobody sends. Do not write faults that merely crash on import,
  and do not try to guess or defeat whatever checks the evaluators use; you have not been
  shown them, and you should not look for them outside the sandbox.
- One fault per patch; small, local changes, as a real bug would be.

## Part 2: legitimate but unusual cases (six)

Two of each kind, as directories `negatives/<name>/` with a `scenario.yaml` and its
fixtures, in the format of `scenarios/` (see `NEGATIVE_FORMAT.md`). The unmodified system
should handle each correctly; they are unusual, not wrong:
- **authorized_master_override:** an authenticated record from the system of record
  corrects a value an earlier notice set, under a deployment that declares that newer
  authenticated records supersede notices.
- **deliberate_version_rejection:** the deployment refuses an interpreter version on
  purpose (for example a candidate model not yet promoted) and has declared that version.
- **valid_late_correction:** a notice is corrected after a plan was released; the
  correction is legitimate and the system should re-plan without losing the earlier view.

## Part 3: the manifest

One file, `manifest.json`, in the format below; nothing else describes the faults:

```json
{"faults": {"01-word.patch": {"family": "unit_calendar", "layer": "L4",
                              "intended_effect": "one or two sentences: what changes and when"}},
 "negatives": {"<dir name>": {"kind": "authorized_master_override",
                              "description": "one or two sentences"}}}
```

`family` is one of the eight names above; `layer` is one of L1, L2, L3, L4, L5, L6,
registry, pipeline, types (the layer whose code you changed). Give the manifest's SHA-256
when you finish. In your final report, list only the number of patches and negative cases
and the manifest's SHA-256: do not describe any fault or case there.
