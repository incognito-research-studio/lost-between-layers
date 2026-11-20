# Lost Between Layers: Replication Package

This package contains the code, data and stored results for the paper

> *Lost Between Layers: Where Runtime Guarantees Break When Language Models Read the Factory.*
> AI4Factory Workshop at AAAI 2027.

The paper asks what a release gate can guarantee when a language model turns supplier, maintenance
and capacity notices into the state that a production planner optimizes over. It states the
condition under which a gate guarantees compliance with the plant's true constraints, and it
tests where that condition is lost between layers: in a replay of stored model replies, in a
fault-injection trial on a reference implementation, and in a conformance test of one
typed-decision reader.

**Every command in this package works from stored data. No model is called and no network is
used.**

## Contents

| Study | Paper | Material in this package |
|---|---|---|
| Reference implementation | Setting and architecture | six layers (L1 interpret, L2 admit, L3 resolve and store, L4 solve, L5 gate, L6 record) in `src/dsoi/` |
| Diagnostic replay | Table 2, Figure 2 | 196 constructed capacity notices with simulated ground truth, and the stored replies of three local open-weight models |
| Fault-injection trial | Table 3 | frozen protocols and check hashes, fault patches with sealed manifests, scenario episodes and legitimate cases for three rounds |
| Typed-decision reader | Table 4 | protocol and stored replies of Laya 0.4.1 on the same 196 notices |

## Requirements

- Python 3.11 or later
- [uv](https://docs.astral.sh/uv/) for the pinned environment (`uv.lock`)

Tested with Python 3.12 on macOS 15 (arm64). The dependencies are numpy, scipy and pyyaml, plus
pytest for the fault-trial harness.

## Reproducing the results

```
uv sync --frozen --extra dev
uv run python tools/reproduce_ai4factory.py            # about 5 minutes
uv run python tools/reproduce_ai4factory.py --quick    # skips the gated round-3 rerun
```

The script reruns each analysis on the stored data and compares its output with the stored
result file: JSON files field by field, other files byte for byte. Tools that update a result
file in place must leave its SHA-256 unchanged. Each step prints `PASS` or `FAIL`, and the
script exits with a non-zero status if any step fails.

| Step | Paper item | Verifies | Does not verify |
|---|---|---|---|
| `e1v2_corrected.py` | Table 2 | rescoring the stored replies through the corrected pipeline reproduces the rates and the case-clustered intervals | the replies themselves |
| `e1v2_violation_sources.py`, `e1v2_log_checks.py`, `e1v2_shadowing.py` | replay text: violation sources, the two log predicates, the first defect's effect | the stated counts | that the declared source policy is the right one |
| `e1v2_abstention_policies.py`, `e1v2_abstention_expectations.py` | Figure 2 | each response to abstention, rescored on the stored replies | behaviour with a real planner or a real confirmation source (q is assumed) |
| `e1v2_abstention_reasons.py` | replay text: abstention split, keyword counts, truth cells, check counts | the stated counts | any cause behind the keyword matches |
| `laya_conformance.py analyse`, `laya_baselines.py` | Table 4 | analysis of the stored Laya replies and the post hoc baselines | the Laya runs themselves |
| `ft_ablation.py` | fault-trial text: 19 of 26, 6 of 9 | detection with check families removed, from stored per-run firings | |
| `ft_r3_holds.py` | fault-trial text: 421 holds, 135, 88, 225 of 286 | the gate's holds, rerun with each round-3 patch applied | |
| `ft_r3_trace.py` | "One escape, traced" | the clean and patched run of fault `06-orchid` on one stored reply | |

**Not reproduced by the script.** The fault-trial detections of Table 3 are read from the stored
`results/fault_trial_*.json` files; the harness that produced them is in `tools/ft_*.py`.
Collecting new model replies requires local models (`src/dsoi/experiments/e1v2_run.py`) or the
Laya package (`tools/laya_requirements.txt`).

## Layout

```
README.md, LICENSE                 this file and the license
MANIFEST.json                      source revision, build time (UTC) and SHA-256 of every file
pyproject.toml, uv.lock            package definition and pinned dependencies
src/dsoi/                          reference implementation and experiment code
configs/, scenarios/               pinned versions and the fixed scenario episodes
data/e1v2/                         constructed notices with truth (cases.jsonl), protocol, truth rules,
                                   deviations
data/fault_trial/                  protocols, freeze records, fault briefs, patches, manifests,
                                   episodes, legitimate cases and sandbox audits for rounds 1 to 3
data/laya/                         protocol, freeze record and deviations of the typed-reader test
results/                           stored replies and every result file the paper reads
tools/                             analysis scripts, fault-trial harness and reproduce_ai4factory.py
```

## Data and provenance

- **Notices and ground truth.** The 196 capacity notices were generated from templates by
  `src/dsoi/experiments/e1v2_notices.py`. The true values come from a simulated world defined by
  the rules in `data/e1v2/TRUTH_RULES.md`; these rules are benchmark assumptions and have not been
  checked by an independent party.
- **Model replies.** Qwen2.5 7B, Llama 3.1 8B and Phi-4 14B were run locally at temperature 0.7,
  with two repeats per notice. The replies are stored in `results/e1v2/replies.jsonl`.
- **Fault patches.** As described in the paper, the faults were written by separate
  language-model agents that had no access to the checks. The protocols, the frozen hashes of the
  checks and scoring, and each round's sealed manifests are in `data/fault_trial/`.
- **Typed-decision reader.** Laya, package `laya` 0.4.1, checkpoint `typed-decisions` at revision
  55cf4c4. Replies are stored in `results/laya_conformance/`.
- **Deviations.** Every departure from a frozen protocol is recorded in the `DEVIATIONS.md` file of
  the study concerned. Frozen files are kept exactly as they were frozen.

## Scope

All evidence comes from one reference implementation and constructed capacity notices. No system
was deployed, and no notice came from a plant. The replay results are a post hoc reanalysis of
development data. The paper's Limitations section gives the full scope.

## Integrity

`MANIFEST.json` records the source revision this package was built from and the SHA-256 of every
file. To check the package:

```
python3 -c "import json,hashlib,pathlib; m=json.load(open('MANIFEST.json')); print([p for p,h in m['sha256'].items() if hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()!=h] or 'all files match')"
```

## License

MIT. See `LICENSE`.

## Citation

```bibtex
@inproceedings{lostbetweenlayers2027,
  title     = {Lost Between Layers: Where Runtime Guarantees Break When Language Models Read the Factory},
  booktitle = {AI4Factory Workshop at the AAAI Conference on Artificial Intelligence},
  year      = {2027}
}
```
