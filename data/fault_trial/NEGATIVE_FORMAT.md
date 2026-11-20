# Format of a negative case

A directory with `scenario.yaml` and any fixture files it names. The file follows
`scenarios/*/scenario.yaml`, with an event list and a `trial:` block that says how the
deployment is configured.

```yaml
id: NEG-...
title: ...
model: capacity_plan            # one of the models in scenarios/ (capacity_plan, lot_allocation, ...)
interpreter_version: "fixture:...@1"   # the version the deployment pins
source_priority: {"email:planner-inbox": 1}   # source ranks by id prefix (higher wins)
world: {...}                    # as in the scenario of the same model
times: {erp_load: 1}
events:                         # in time order
  - {id: msg-1, t: 2, kind: message, source: "email:...", text: "...", fixture: fixtures/a.json}
  - {id: rec-1, t: 3, kind: record, source: "erp:capacity-master",
     ops: [{op: assert, subject: "plant:X", predicate: capacity, value: 650,
            unit: "units/week", valid_from: 42, valid_to: 45}]}
  - {t: 3.5, kind: approval, approval_id: apr-1, approval_kind: lot-release,
     variable: "x[L1]", approver: qa.manager, source: "idp:qa-approvals", expires_at: 10}
  - {t: 4, kind: sor, values: {"plant:X|capacity|42": 600}}   # system of record changes
  - {t: 5, kind: decide}
trial:
  gate: nominal                 # nominal | box_robust | vertex | recheck
  resolution: span              # optional L3 conflict policy: latest | priority | span | block
  supersession: none            # none | authenticated_latest
  threshold_rules: [{kind: lot-release, variable: "x[L1]", approvers: [qa.manager]}]
  declared_versions: []         # interpreter versions the deployment declared and refuses
  system_of_record: {}          # initial re-read values for the recheck gate
```

A fixture is the recorded interpreter output for one message, as in
`scenarios/s1_capacity_cut/fixtures/interpret_correct.json`; its `interpreter_version` is
the version that answered. An approval event is authenticated unless `authenticated: false`.
