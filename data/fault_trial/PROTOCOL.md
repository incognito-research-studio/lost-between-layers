# Fault-injection evaluation with separated roles: protocol

Written by Claude (the architecture role), 8 November 2026, after the sixth external review
(experiment 1). Frozen with the instruments listed below before any fault was written. It
replaces `data/e1v2/FAULT_TRIAL.md` (4 November), whose harness, `tools/fault_trial.py`,
stays in the artifact unchanged for the reviewer check.

## Question

Do explicit contracts, provenance and frozen checks detect, before or after execution,
integration faults that the people and agents who wrote the checks did not write, beyond
what solver feasibility checks and generic validation detect? The trial does not test
operational reliability.

## Roles and what each could see

| Role | Who | Saw | Did not see |
|---|---|---|---|
| Architecture | Claude Code (Claude Opus 5.5), the session that wrote the pipeline, the checks and this protocol | everything | the faults, the negative cases and the manifest, until scoring |
| Fault generation | a separate Claude agent started with no conversation history, working in a sandbox copy (`tools/ft_sandbox.py`) | the pipeline source except the files below, the layer contracts (Table 2 of the paper), the scenario files and formats, `FAULT_BRIEF.md` | `src/dsoi/properties.py`, `src/dsoi/layers/l5_monitors.py`, `src/dsoi/trace.py` (a walk-through that calls the predicates), everything under `tools/` and `tests/`, `results/`, the paper's checkability table and Section 10.3, the review files, this protocol's check list |
| Evaluation | `tools/ft_trial.py`, then `tools/ft_score.py` | the patches and negative cases; the manifest only in `ft_score.py`, after the results are written, and only if its SHA-256 matches the one recorded before the run | |

The separation is by sandbox and by instruction. The generator's tool calls are kept and
are checked afterwards for any read outside the sandbox; the result of that check is
reported. Separate agents of one model family reduce information leakage; they are not
independent scientific validation. The trial is reported as an "AI-generated
fault-injection evaluation with separated roles", not as a blind human trial. A coauthor
who did not write the checks rates the realism of a sample of patches if one is available.

## Frozen before any fault exists

`tools/ft_harness.py` (worlds, checks, configurations, layer and timing tags, the effect
trace), `tools/ft_trial.py` (classification), `tools/ft_score.py` (metrics),
`data/fault_trial/authority.yaml` (the declared authority policy of the e1v2 world),
`src/dsoi/properties.py` (predicates), this file and `FAULT_BRIEF.md`, at the commit that
adds them; `results/fault_trial_v2.json` records their SHA-256. Two of the 4 November
monitors were made policy-aware before freezing, after the review asked for exactly that
(Section 10.3 already says both read against declared policy): `precedence` replaces
`shadowed`, comparing each fact a decision read with the fact a declared authority policy,
kept as data, selects; `version` flags a version refusal only when the served version is not
one the deployment declared. Both changes were made before any fault or negative case
existed.

## Worlds

- **e1v2:** the corrected replay of the stored e1v2 replies, test part, three models, the
  widened box gate at the stored margins (672 runs).
- **scenarios:** twelve fixed episodes of the scenario suite (S1 nominal, box and recheck
  with the system of record changing between two decisions; S2 with a threshold rule and an
  approval; S3 recheck and nominal; S4 span and priority; S5; S6; S7; S8 vertex).
- **negatives:** the generator's legitimate but unusual cases, on the unmodified pipeline.

## Configurations (nested)

1. **solver:** the released action satisfies the gate's feasible set (K-prop).
2. **generic:** plus exceptions, sanity of committed facts (finite, non-negative values
   within their interval, non-empty valid time), sanity of plans and executions, and
   required record fields.
3. **full:** plus the architecture's checks: G1, R2, C1, audit completeness, G2,
   execution fidelity, declared precedence, declared versions, committed-but-unread,
   C2, replay from the store, restart round trip with replay without live reads, and L2
   admission of threshold actions.

Each check is tagged with the layers it points to and as pre-execution (it can be
evaluated from what is known before execution) or post-execution. G2 and
committed-but-unread can fire on correct runs; for them, and for faults generally, only a
firing that the clean run does not show counts.

## Classification and scoring (fixed)

Per patch: rejected (does not apply, or touches a file outside `ALLOWED`), broken (the
patched package does not import or the harness fails), inert (every run's effect trace and
outcomes equal the clean run's), or effective. Per effective patch and configuration:
detected, detected before execution, layers implicated, silent-violation runs (more
violations than clean, no new firing), releases a pre-execution check would hold. Negative
cases: a false alarm is any non-relative check of the configuration firing.

Unit of analysis: the fault. Reported per configuration and fault family: detection rate
(Wilson 95%), pre-execution detection, faults with silent violations, localization
accuracy, false alarms on negatives, release counts, and exact McNemar tests between
configurations on the same effective faults. Inert, broken and rejected patches are
reported, not dropped. After unblinding, each effective patch's observed change is compared
with the manifest's intended effect by a reader; a mismatch is reported.

## Limits

One harness, two small worlds, stored replies and fixture interpreters; faults and negative
cases written by one agent of the same model family as the architecture role; detection here
does not show that operational faults would be found.
