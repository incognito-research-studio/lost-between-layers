# Typed-reader conformance with Laya: protocol

Frozen 15 Nov 2026 before the first run on the study notices (`FREEZE.json` holds the SHA-256 of
this file, of `tools/laya_conformance.py`, of `tools/laya_requirements.txt` and of the notices).
`tools/laya_conformance.py run` refuses to start if any of them differs.

## Question

Does a typed-decision model meet the obligations that Section 6 ("Typed-decision readers") and
Remark 1 place on L1? The study measures contract properties. It does not measure reading
accuracy or the value of reading, and it compares Laya with no other reader.

## Reader

Laya, ConvAI Innovations, Apache-2.0 weights: package `laya==0.4.1`, repository
`convaiinnovations/laya`, subfolder `typed-decisions`, revision
`55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851` (the revision `laya` 0.4.1 pins). Zero-shot, with the
checkpoint's own configuration: no fine-tuning, no calibration and no change of the token budget.
Environment: `tools/laya_requirements.txt`, Python 3.12, offline (`HF_HUB_OFFLINE=1`).

## Inputs

The 196 constructed notices of `data/e1v2/cases.jsonl` (154 change, 28 no change, 14 vague),
both parts, since nothing is fitted. The state passed to Laya names the plants ("Plant X is the
Northfield plant. Plant Y is the Riverside plant."), the planning weeks (40 to 49) and each
message in arrival order, without its source identifier.

## Questions and candidates

The candidates are built before the reader runs, from the planning horizon alone.

- `start`: Choice over "no change" (the value in force) and the ten weeks 40 to 49, each with a
  one-sentence definition ("The change to Plant X's capacity begins in week w.").
- `end`: Choice over the ten weeks ("... ends in week w.").
- `cant_tell`: companion Noul, "Is it impossible to tell from the messages which weeks of Plant
  X's weekly capacity change?". A probability above 0.5 maps to abstention.
- `wide56`: one Choice over "no change" and all 55 contiguous ranges, asked once per case at the
  canonical order, in its own call. Only Laya's report of how many options stayed
  distinguishable within its question budget is analysed. A probe on an invented notice, before
  freezing, showed that this question exceeds the budget; this is why `start` and `end` replace
  it.

Every contiguous range of planning weeks is a (start, end) pair, so construction (Remark 1 (i))
is complete by design for every change and no-change notice; it is reported, not tested.

## Conditions

Per case, on CPU: the canonical order (`base`); the reversed order; five shuffles (seeds 0-4,
drawn per case and question from `numpy.random.default_rng([20261115, seed, h])`, where h is
taken from the SHA-256 of the case id and question); neutral names ("option 01" ...,
assigned by a seeded permutation, definitions unchanged); and exchanged names (a seeded
derangement binds each name to another candidate's definition). The `base` condition is
repeated once on CPU and once on the Apple GPU (MPS).

## Measures and decision rules

All comparisons are by the candidate whose definition was chosen, not by the returned name.

1. Repeatability: replies whose probabilities (as Laya returns them, four decimals) are
   identical between the two CPU runs; same for CPU against MPS, plus decisions that agree
   and the largest probability difference.
2. Order: cases whose `start` or `end` choice changes under the reversed order and under each
   shuffle, cases that change under any of the six orders, mean total variation.
3. Names: cases that change under neutral names and under exchanged names; under exchanged
   names, cases that returned the same name as at `base` although it now carries another
   definition (the choice followed the name).
4. Null: `cant_tell` above 0.5, by notice class.
5. Remark 1: construction (i); retention (ii) by the top choices (both right; for no-change
   notices `start` = "no change" alone) and by the smallest set of probability mass 0.9 per
   question; the mean number of the 56 candidates those sets allow.

A property is reported as met only if it holds on every case. Otherwise the paper reports the
measured count and states the L1 obligation it implies. No threshold, candidate, wording or
condition is changed after the first run; any deviation is logged in `DEVIATIONS.md` in this
folder with its reason, and the frozen result is reported alongside.
