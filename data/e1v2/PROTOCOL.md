# E1 v2 protocol (frozen before any model call)

This file, `configs/e1v2.yaml`, the case set `data/e1v2/cases.jsonl` and the code under
`src/dsoi` are hashed into `data/e1v2/manifest.json` by `dsoi-e1v2 freeze` before any
interpreter is run. Every stored reply carries the manifest hash in force when it was
requested, and the summary records the hash in force when it was scored; a difference is
reported as a deviation, file by file.

## Question

Can a language-model interpreter produce uncertainty sets that cover the true state of
operational notices (the coverage goal of the assurance pattern), and what do the gates
then release, at what violation rate and cost?

## Cases

196 constructed capacity notices for two plants (`src/dsoi/experiments/e1v2_notices.py`),
14 per category: hedged, exact, "up to" and "at least" percentages; absolute and per-day
figures; a relative date scope; a notice about the other plant only; one about both plants;
a planner estimate followed by a supplier portal figure on the same weeks (two messages);
an injected approval with a real change; an in-message correction; an out-of-scope notice;
a vague notice. The notices are constructed, not collected; the truth of each phrasing is
defined in the generator before any run, and coverage is measured against it. Nobody
outside the author team has reviewed the wording or the truth definitions.

Split: a seeded, stratified draw puts 40% of each category (84 cases) in a calibration part
and 60% (112 cases) in a test part. Every reported rate is on the test part.

## Interpreters

Three local open-weight models from three families, served by Ollama on one machine
(qwen2.5:7b-instruct, llama3.1:8b, phi4), temperature 0.7, two repeats per case with
seeds derived from the case id and the repeat, four cases sent to the server at once. All use the same system prompt and output
schema (`src/dsoi/layers/l1_claude.py`) and pass the same L1 contract checks. Before each
message, the interpreter receives a text snapshot of the current state: plant names,
nominal capacity, units, and any capacity already recorded.

Claude models (added on 23 Oct 2026 after the authors approved a $25 budget; see the
manifest history): claude-haiku-5-5, claude-sonnet-5-5 and claude-opus-5-5, effort
`medium`, at most 4,096 output tokens, two repeats, the same system prompt, user prompt,
snapshot and output schema as the local models. They run through the Message Batches API,
in rounds (a case's later message waits for the reply to the earlier one), with the shared
prefix of each request marked for prompt caching; caching and batching change the price,
not the text the model reads. No fallback model is configured, so each model is measured as
itself and a refusal is an abstention. The API has no sampling seed, so repeats measure
variability without reproducing it. A batch is sent only if its worst case (every request
at the full output limit) fits in the budget left, and the ledger records what each batch
cost. The case generator and this harness were written with an AI assistant (Claude); the
models measured received only the notices, the snapshot and the instructions.

## Pipeline and arms

Each case runs through the reference implementation: L1, L2, L3 (span policy, supplier
portal ranked above planner e-mail), L4 (the capacity plan for plant X) and four arms at L5:
point (nominal gate), stated (box-robust gate on the stated intervals), calibrated
(box-robust gate on the stated intervals widened by the model's split-conformal margin q)
and recheck (system-of-record recheck, a reference under A-ORACLE by construction).

## Measures (definitions in `src/dsoi/experiments/e1v2_score.py`)

Joint coverage of plant X's capacity in every week (stated and calibrated); scope exact;
entity correct; quiet notices left quiet; abstention; per arm: release rate, violations
per case (Pr[V]), violations among released cases (Pr[V | R]) and mean shortfall above
the true optimum when released; variability of the committed state across repeats. Rates
with 95% Wilson intervals. The calibration margin q is fitted per model on the calibration
part at alpha = 0.1 and applied unchanged to the test part.

## Pilot before the freeze

Before the freeze, six calibration cases (hedged, conflict, two plants, relative scope)
were sent once to each local model to check the plumbing and the run time; those replies
are not part of the study. The pilot led to three changes, all made before the freeze:
the `reason` field of the output schema is limited to 300 characters for the local
models (qwen2.5 filled the token budget with it and was cut off), the snapshot now states
that plant names map to the subject identifiers and how week ranges map to valid time
(phi4 abstained on a named plant despite the mapping), and repeats were reduced from three
to two, with four cases in parallel, to keep the run to a few hours. The pilot showed
frequent abstention by the small models; nothing was changed to reduce it.

## Changes after the first freeze

On 23 Oct 2026 the protocol was frozen again (manifest history) after qwen2.5 had finished:
the Claude backend was added (`l1_claude.build_cached_params`, the Claude section of
`e1v2_run.py`, `run-claude` in `e1v2_cli.py`, the `claude` block of the config), and the
local run continued with one case at a time instead of four, because the machine running
the models overheated. Neither change touches the cases, the prompts or the scoring code;
the replies of qwen2.5 carry the first manifest hash, all later replies the second.

## Exclusions and deviations

No reply is excluded. Failed calls, malformed output, refusals and truncations are scored
as what the L1 contract makes of them (abstentions). Any change to a frozen file after the
freeze is listed in the summary as a deviation; nothing is re-run silently.
