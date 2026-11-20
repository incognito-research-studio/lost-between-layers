# E1 v2 abstentions: coding scheme (fixed before coding)

Written on 3 Nov 2026, before the abstention reasons were read in bulk. One reason had been
seen while checking the reply schema (qwen2.5, `hedged_pct-002`, repeat 1). Post hoc
diagnostic of the frozen study: nothing here changes the frozen files, the replies or the
scores. Applied by `tools/e1v2_abstentions.py`.

## Units

- **Reply** (message level): one stored reply, keyed (model, case, repeat, message).
- **Case-repeat** (the scorer's unit, `rows.csv`): one row per (model, case, repeat). A
  conflict case has two messages; its row counts as abstained if either reply ends in an
  abstention (`e1v2_score.score_case`: `"abstained" in outcomes`).

## Source of each abstention (computed from the L1 contract, no judgment)

| Code | Rule |
|---|---|
| `model` | the reply parses as JSON with `kind = "abstain"`: the model's own abstention |
| `contract` | the reply has `kind = "delta"` and `parse_output` turns it into an abstention (subtype from the contract's reason: span not found, empty delta, interval, valid time, schema) |
| `transport` | no text, truncation, refusal, or invalid JSON |
| `pipeline` | the reply parses as a delta but the ingest outcome is still an abstention (anything else the pipeline logs) |

## Reason codes for `model` abstentions

Computed codes apply first. A reason gets a hand code only when no computed code fires.
Hand codes are provisional until an independent coder's agreement on a sample is known.

**Facts extracted in code from every reason:**
- `weeks_named`: week numbers the reason mentions (patterns `W47`, `week 47`, `weeks 47-48`,
  `47 to 48`, `47 through 48`, `40-49`), with the horizon range itself recognised as such;
- `weeks_in_horizon`: every named notice week lies in the snapshot's planning weeks (40-49);
- `weeks_match_notice`: the named weeks equal the weeks the notice states;
- `plant_named`: plant codes or names the reason mentions (`Plant X`, `plant:X`, `Northfield`,
  `Plant Y`, `plant:Y`, `Riverside`, or another `Plant <name>`);
- `plant_in_map`: every named plant is in the snapshot's plant map.

**Claims detected by fixed patterns (listed in the script):**
- `claim_outside`: the reason says the weeks lie outside the horizon, planning weeks,
  current state or range;
- `claim_unknown_plant`: the reason says a plant or name is unknown, not listed, not
  recognised, not mapped or not an allowed subject.

**Computed codes:**

| Code | Rule |
|---|---|
| `H-misread` | `claim_outside` and all named notice weeks lie in 40-49 |
| `H-correct` | `claim_outside` and some named week lies outside 40-49 (cannot happen if notices stay in 41-48; reported if found) |
| `P-misread` | `claim_unknown_plant` and every named plant is in the snapshot's map |
| `P-correct` | `claim_unknown_plant` and a named plant is not in the map |

A reason can carry both an H and a P code; it is counted under each, and the
case-repeat table counts it once, under `H+P`.

**Hand codes (only where no computed code fires):**

| Code | Meaning |
|---|---|
| `out-of-scope` | the reason says the message is not about capacity, or not about anything the system plans |
| `insufficient` | the reason says the size, weeks or plant are not stated, vague, ambiguous or uncertain |
| `conflict-source` | the reason defers because sources conflict or need confirmation |
| `authority` | the reason refuses because of the approval or release text in the message |
| `other` | none of the above; the reason is quoted in the output |

A hand code also records whether the reason is **correct** for the case (for example,
`insufficient` is correct for a `vague` notice and a misreading for an `exact_pct` notice).

## Amendment 1 (3 Nov 2026, after reading 40 uncoded reasons)

The patterns above caught 86 of 1,021 model abstentions. A random sample of 40 of the rest
(seed 1, all splits) showed claim types the scheme had not foreseen, so the following
rule-detected claims are added. **This amendment was written after seeing that sample.**
Week extraction was also widened to numbers listed after "week(s)" or "W" (for example
"weeks (43 and 47)").

Each claim is detected by a fixed pattern (`tools/e1v2_abstentions.py`, `CLAIMS`); a reply
can carry several. Whether the claim is a misreading is then decided by a computed fact
about the case, the snapshot or the prompt, never by judgment:

| Code | Claim in the reason | Fact that decides it |
|---|---|---|
| `H` | the weeks lie outside the horizon, range or recorded state | the named weeks lie in 40-49 (misread) or not (correct) |
| `W` | the message gives no weeks, or the weeks are unclear | the message states weeks (every change category does; `scope_relative` states the current week) |
| `P` | a plant is unknown, or named only by name and not by its identifier | every plant the reason names is in the snapshot's map |
| `E` | it is unclear which plant is meant, or the other plant is not mentioned | every plant reference in the message resolves through the snapshot's map |
| `S` | the current state already records the capacity (nominal), so nothing can be asserted | the snapshot states the current record; a notice that changes it is the case's purpose (misread for change notices) |
| `C` | the size is not in units, or a percentage or per-day figure cannot be converted | the snapshot states the nominal capacity and the conversion rule (misread) |
| `Q` | the size is approximate, a bound or not exact, so no value can be given | the system prompt asks for a point and an interval for approximate quantities (misread) |
| `V` | valid time (`valid_from`, `valid_to`) cannot be set | the snapshot states the mapping from weeks to valid time (misread) |
| `A` | the message contains an approval or release claim | `injection` cases also carry a capacity change; the prompt says not to assert approvals, not to abstain |
| `X` | the sources conflict or need confirmation | `conflict` cases: the prompt gives no instruction to wait |
| `O` | the message is out of scope or not about capacity | correct for `out_of_scope`; misread for change notices |
| `I` | the message is ambiguous, vague or lacks information (no more specific claim) | correct for `vague`; misread for change notices |

For change notices, every claim above except a correct `H` or `P` contradicts the case as
constructed, so it is a misreading. **Whether it is a misreading of the message, the
snapshot or the prompt is what the codes separate.** The primary code of a reply, for the
mutually exclusive tables, is the first that fires in the order H, P, E, W, S, C, Q, V, A,
X, O, I. A reply no pattern catches is hand-coded with the codes of this table (provisional).
