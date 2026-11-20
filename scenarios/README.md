# Scenarios (Table IV)

Each folder holds one ATAM-style scenario: `scenario.yaml` (decision model,
world, timed events, stimulus, required response, analytic measure,
dependencies, risk) and recorded interpreter outputs in `fixtures/`.

| ID | Folder | Model | Test |
|----|--------|-------|------|
| S1 | `s1_capacity_cut` | capacity_plan | `tests/scenarios/test_s1.py`, `dsoi-trace S1` |
| S2 | `s2_quality_hold` | lot_allocation | `tests/scenarios/test_s2_quality_hold.py` |
| S3 | `s3_supplier_suspension` | supplier_mix | `tests/scenarios/test_s3_supplier_suspension.py` |
| S4 | `s4_contradictory_notices` | capacity_plan | `tests/scenarios/test_s4_contradictory_notices.py` |
| S5 | `s5_injected_approval` | lot_allocation | `tests/scenarios/test_s5_injected_approval.py` |
| S6 | `s6_model_update` | capacity_plan | `tests/scenarios/test_s6_model_update.py` |
| S7 | `s7_late_correction` | capacity_plan | `tests/scenarios/test_s7_late_correction.py` |
| S8 | `s8_pallet_conversion` | warehouse_flow | `tests/scenarios/test_s8_pallet_conversion.py` (artifact extension: coefficient uncertainty) |

## Events

S1 uses a single `stimulus`. The others list timed `events`, run by
`dsoi.scenarios.run_events`:

```yaml
events:
  - {id: msg-hold, t: 2, kind: message, source: "email:qa-inbox",
     text: "QA notice: quality hold on lot L1 pending inspection.",
     fixture: fixtures/hold.json}                       # through L1 -> L2 -> L3
  - {t: 3, kind: decide}                                 # L3 -> L4 -> L5 -> L6
  - {id: qa-approval, t: 6, kind: record, source: "idp:qa-approvals",
     ops: [{op: supersede, subject: "lot:L1", predicate: hold, value: 0,
            valid_from: 42, valid_to: 43}]}              # authenticated, enters at L2
```

`run_events` can swap a fixture, move an event in time, or supply a generated
interpreter output per event id; tests and simulations use these to run variants.

## Fixtures

```json
{
  "interpreter_version": "fixture:s1-correct@1",
  "output": {"kind": "delta", "ops": [{
    "op": "supersede", "subject": "plant:X", "predicate": "capacity",
    "value": 700, "lo": 650, "hi": 750, "unit": "units/week",
    "valid_from": 42, "valid_to": 45,
    "span_text": "capacity reduced by about 30% for weeks 42-44"}]}
}
```

`span_text` must occur verbatim in the message; the L1 contract turns any
violation into an abstention. An abstention is written as
`{"kind": "abstain", "reason": "..."}`.

A hedged reading (Sec. VI) puts an interval around its point. S1's doubtful
week 44 is `"value": 1000, "lo": 650, "hi": 1000`; S3's hedged suspension is
`"value": 1, "lo": 0, "hi": 1`. Give a hold or suspension the restrictive point:
L2 refuses an open point (0) from text as a lift.

One YAML caution: a bare `true:` key parses as the boolean True, so the
ground-truth block in a world is named `truth:`.
