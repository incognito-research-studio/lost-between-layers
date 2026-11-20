"""dsoi-e1: run experiment E1 (live interpreter calibration and the gate ladder).

  generate          write the notice set to data/e1/notices.jsonl
  dryrun            mock replies -> score (no API calls, no cost)
  pilot --yes       one live call per pilot notice and model (spends money)
  plan              size the main run to the remaining budget from the pilot
  run --yes         submit the planned batches and collect them (spends money)
  collect ID MODEL  resume collection of a submitted batch
  score             score stored replies (offline)
  status            budget ledger and stored replies

API credentials come from the environment or a git-ignored .env file
(ANTHROPIC_API_KEY=...). Never paste a key into chat.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

from dsoi.experiments import e1_harness as H
from dsoi.experiments.e1_mock import dry_run_records
from dsoi.experiments.e1_notices import CATEGORIES, generate, load, save
from dsoi.experiments.e1_score import score, world_scenario

NOTICES = Path("data/e1/notices.jsonl")
RESULTS = Path("results/e1")


def _config(path: Path) -> dict:
    return yaml.safe_load(path.read_text())


def _notices(cfg: dict) -> list:
    if not NOTICES.exists():
        save(generate(cfg["notices"]["n_per_category"], cfg["seed"]), NOTICES)
    return load(NOTICES)


def _client():
    H.load_env()
    try:
        import anthropic
    except ImportError:
        sys.exit("install the providers extra: uv sync --extra providers")
    return anthropic.Anthropic()


def _print_summary(summary: dict) -> None:
    keys = ["replies", "abstained", "no_change_correct", "scope_exact", "coverage_A2",
            "unsafe_miss", "point_abs_error", "interval_width", "violations_nominal",
            "violations_box_robust", "violations_state_vertex", "violations_recheck",
            "box_attribution_exceptions", "state_varies_across_repeats", "cost_usd"]
    models = list(summary["models"])
    print(f"{'':<28}" + "".join(f"{m:>20}" for m in models))
    for k in keys:
        cells = []
        for m in models:
            v = summary["models"][m].get(k)
            cells.append("-" if v is None else (f"{v:.3f}" if isinstance(v, float) else str(v)))
        print(f"{k:<28}" + "".join(f"{c:>20}" for c in cells))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="dsoi-e1", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["generate", "dryrun", "pilot", "plan", "run",
                                        "collect", "score", "status"])
    ap.add_argument("args", nargs="*")
    ap.add_argument("--config", type=Path, default=Path("configs/e1.yaml"))
    ap.add_argument("--yes", action="store_true", help="confirm commands that spend money")
    ap.add_argument("--phase", default="main", help="score: pilot | main | all")
    a = ap.parse_args(argv)
    cfg = _config(a.config)
    store, ledger_path = RESULTS / "responses.jsonl", RESULTS / "ledger.json"
    schema = world_scenario(cfg["world"]).schema()

    if a.command == "generate":
        notices = generate(cfg["notices"]["n_per_category"], cfg["seed"])
        save(notices, NOTICES)
        print(f"wrote {len(notices)} notices to {NOTICES}")
        return 0

    notices = _notices(cfg)
    by_id = {n.id: n for n in notices}

    if a.command == "dryrun":
        records = dry_run_records(notices, cfg["repeats"], cfg["seed"])
        summary = score(records, notices, cfg["world"], RESULTS.parent / "e1-dryrun")
        _print_summary(summary)
        print(f"\nwrote {RESULTS.parent / 'e1-dryrun'}/scores.csv and summary.json")
        return 0

    ledger = H.Ledger.open(ledger_path, cfg["budget_usd"])

    if a.command == "status":
        recs = H.load_records(store)
        print(f"budget ${ledger.budget:.2f}  spent ${ledger.spent:.4f}  "
              f"remaining ${ledger.remaining:.4f}")
        for phase in ("pilot", "main"):
            n = sum(r["phase"] == phase for r in recs)
            print(f"{phase:<6} replies stored: {n}")
        for b in ledger.batches:
            print(f"batch {b['id']}  {b['model']}  n={b['n']}  estimate ${b['estimate']}")
        return 0

    if a.command == "score":
        phase = None if a.phase == "all" else a.phase
        summary = score(H.load_records(store), notices, cfg["world"],
                        RESULTS / f"score_{a.phase}", phase)
        _print_summary(summary)
        return 0

    if a.command == "plan":
        pilot = [r for r in H.load_records(store) if r["phase"] == "pilot"]
        plan = H.plan_main(cfg, notices, pilot, ledger)
        print(json.dumps({k: v for k, v in plan.items() if k != "notice_ids"}, indent=2))
        print(f"notices in plan: {len(plan['notice_ids'])}")
        return 0

    if not a.yes:
        sys.exit(f"`{a.command}` spends money; re-run with --yes "
                 f"(budget ${ledger.budget:.2f}, spent ${ledger.spent:.4f})")
    client = _client()

    if a.command == "pilot":
        picks = [next(n for n in notices if n.category == c) for c in CATEGORIES]
        picks = picks[:cfg["pilot"]["n_notices"]]
        recs = H.run_pilot(client, cfg, picks, schema, ledger, store)
        for model in cfg["models"]:
            mine = [r for r in recs if r["model"] == model]
            out = sum(r["usage"]["output_tokens"] for r in mine) / max(len(mine), 1)
            cost = sum(r["cost_usd"] for r in mine) / max(len(mine), 1)
            print(f"{model:<20} mean output tokens {out:7.0f}  mean cost ${cost:.5f}/request")
        print(f"spent ${ledger.spent:.4f} of ${ledger.budget:.2f}")
        plan = H.plan_main(cfg, notices, [r for r in H.load_records(store)
                                          if r["phase"] == "pilot"], ledger)
        print(f"main-run plan: {len(plan['notice_ids'])} notices x {plan['repeats']} repeats, "
              f"estimate ${plan['estimate_usd']:.2f} (with safety factor)")
        return 0

    if a.command == "run":
        pilot = [r for r in H.load_records(store) if r["phase"] == "pilot"]
        plan = H.plan_main(cfg, notices, pilot, ledger)
        H.submit_and_collect(client, cfg, plan, by_id, schema, ledger, store)
        print(f"done; spent ${ledger.spent:.4f} of ${ledger.budget:.2f}")
        return 0

    if a.command == "collect":
        batch_id, model = a.args
        H.collect_batch(client, batch_id, model, by_id, cfg, ledger, store)
        print(f"collected {batch_id}; spent ${ledger.spent:.4f}")
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
