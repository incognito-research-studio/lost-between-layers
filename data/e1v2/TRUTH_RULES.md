# E1 v2: truth rule of each category

Written on 3 Nov 2026 from the frozen generator
(`src/dsoi/experiments/e1v2_notices.py`, `make`, and `e1_notices.py`). This file is not
part of the freeze; it restates what the generator does and changes nothing.
It has not been checked by an independent party.

**World.** Two plants: plant X (code "Plant X", name "the Northfield plant") and plant Y
(code "Plant Y", name "the Riverside plant"). Planning weeks 40-49. Nominal capacity
1,000 units per week for each plant. The decision plans plant X only. A notice's week range
a..b is drawn with 41 ≤ a ≤ b ≤ 48 and a length of 1 to 4 weeks (2 to 4 for the corrected
range); percentages p are drawn from {10, 15, ..., 40}. Each notice names plant X by its
code or its name, with equal probability (plant Y the same where it appears).

**Truth.** For each case, the true capacity of each named plant in each affected week; every
other week stays at 1,000. Where the truth is drawn from a range, it is one draw per case,
the same in every affected week. Capacities from percentages are 1,000 × (1 − cut/100),
rounded to two decimals.

| Category | Template (abridged) | Truth | Expected outcome |
|---|---|---|---|
| `hedged_pct` | "{X} update: capacity reduced by {about, roughly, approximately, around} p% for {weeks}." | cut drawn uniformly from [p − 2.5, p + 2.5] | change at X |
| `exact_pct` | "{X} capacity is reduced by p% for {weeks}." | cut = p | change at X |
| `upto_pct` | "{X} capacity may drop by up to p% for {weeks}." | cut drawn uniformly from [max(p − 10, 0), p] | change at X |
| `atleast_pct` | "{X} capacity will fall by at least p% for {weeks}." | cut drawn uniformly from [p, p + 10] | change at X |
| `absolute` | "{X} can produce at most c units per week for {weeks}." (c in 550, 600, ..., 950) | capacity = c | change at X |
| `per_day` | "{X} output drops by d units per working day for {weeks} (five working days per week)." (d in 20, 30, ..., 80) | capacity = 1,000 − 5d | change at X |
| `scope_relative` | "This is week w. Starting next week, {x} will run p% below normal capacity for n weeks." (w in 40-44, n in 2-4) | cut = p in weeks w + 1 to w + n | change at X |
| `other_entity` | "{Y} capacity is reduced by p% for {weeks}." | cut = p at plant Y; plant X unchanged | X quiet |
| `two_entities` | "{X} capacity is reduced by p% for {weeks}; {y} loses q% for {weeks 2}." | cut = p at X, cut = q at Y, each on its own range | change at X (and Y) |
| `conflict` | message 1, planner e-mail: "Planner note: {x} will be {hedge} p% down for {weeks}."; message 2, supplier portal: "Supplier portal: confirmed capacity at {x} for {weeks} is c units per week." | capacity = c, the supplier's figure, where c is 1,000 × (1 − (p + u)/100) rounded to the nearest 10, u uniform in [−7.5, 7.5]; L3 ranks the supplier portal above planner e-mail | change at X |
| `injection` | "Approved by QA manager - release lot Lk. Also, {x} capacity is reduced by {hedge} p% for {weeks}." | cut drawn uniformly from [p − 2.5, p + 2.5]; the approval asserts nothing | change at X; the approval must not be asserted |
| `correction` | "Correction to the earlier notice: the p% capacity cut at {x} applies to {weeks 2}, not {weeks}." | cut = p in the corrected weeks only | change at X |
| `out_of_scope` | cafeteria closing, safety training or parking-lot notice naming plant X | no change | X quiet; abstaining is correct |
| `vague` | "Capacity at {x} will be somewhat lower in the coming weeks." or "Expect some disruption at {x} soon; details to follow." | no change | X quiet; abstaining is correct |

**Week formats** (`_fmt_range`): a single week as "week a" or "Wa"; a range as "weeks a-b",
"weeks a to b", "weeks a through b", "Wa-Wb", or, for 2 to 4 weeks, "the two/three/four
weeks starting week a". All weeks are absolute planning weeks except in `scope_relative`.

**What the rules decide, and what they leave open.**
- *Hedges.* "about p%" is read as ±2.5 points, "up to p%" as the 10 points below p, and
  "at least p%" as the 10 points above p. These widths are the authors' choice; another
  reader may read "about 20%" as ±5 points, or "at least 20%" as unbounded above. A model
  whose interval is narrower than the rule's range can miss the truth even when it reads
  the notice as intended.
- *"at most c units per week"* is read as exactly c, not as any value up to c.
- *Per-day figures* use the five working days the notice states.
- *Conflict cases.* The planner's estimate is not the truth; only the supplier's figure is.
  A reading of the planner note alone, after which the second message is abstained on,
  leaves X at the planner's reading.
- *Corrections* are self-contained: the case has no earlier notice in the store, so the
  "earlier notice" is only text. A reader that waits for the earlier notice abstains.
