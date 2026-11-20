# Deviations from the frozen protocol

1. 15 Nov 2026, after the run, presentation only. The table is written by `tools/laya_table.py`
   instead of the `table` step of the frozen `tools/laya_conformance.py`, which pooled change
   and no-change notices in the retention rows (hiding that "no change" was never retained) and
   printed a one-value range as "20 to 20". No measure, threshold or reply changed;
   `results/laya_conformance.json` is the output of the frozen `analyse` step.

# Post hoc analyses (not in the frozen protocol)

1. 16 Nov 2026, after the run.
   `tools/laya_baselines.py` adds reference points that the protocol did not specify: random
   sets of the same size as the 0.9-mass sets, a notice-blind prior (the 24 ranges most frequent
   among the other change notices, leave-one-out), a uniform guess for the top choice, order
   flips and name-following by the base reply's top-two margin, and whether the 0.9-mass set
   changes with order. It reads the stored replies of the first CPU run and writes
   `results/laya_baselines.json`; no frozen file, reply, measure or threshold changed, and no
   model was called. The notice-blind prior retains the truth in 151 of 154 change notices,
   more than the reader's 135, because the notices use only 26 distinct true ranges; retention
   in this benchmark therefore does not show that the reader used the notice.
