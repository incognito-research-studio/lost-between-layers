"""dsoi-e1v2: the pre-registered live-interpreter study (data/e1v2/PROTOCOL.md).

  generate        write the case set to data/e1v2/cases.jsonl
  freeze          hash the protocol, config, cases and code into data/e1v2/manifest.json
                  (once; a later freeze needs --reason and is kept in the manifest history)
  dryrun          mock interpreter -> score, into results/e1v2-dryrun (no model calls)
  run             local models through Ollama; refuses if a frozen file has changed
  run-claude      Claude models through the Message Batches API (spends money: needs --yes;
                  --pilot sends six calibration cases once, outside the study's store)
  score           score the stored replies into results/e1v2 (offline, repeatable)
  status          stored replies per model, and whether the freeze still holds
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import yaml

from dsoi.experiments.e1v2_notices import generate, load, save
from dsoi.experiments.e1v2_run import (
    ClaudeBatch,
    MockBackend,
    OllamaBackend,
    load_replies,
    run,
    run_claude,
)
from dsoi.experiments.e1v2_score import score

CONFIG = Path("configs/e1v2.yaml")
CASES = Path("data/e1v2/cases.jsonl")
MANIFEST = Path("data/e1v2/manifest.json")
RESULTS = Path("results/e1v2")
REPLIES = RESULTS / "replies.jsonl"
FIXED = ["configs/e1v2.yaml", "configs/versions.yaml", "data/e1v2/PROTOCOL.md",
         "data/e1v2/cases.jsonl"]


def frozen_files(root: Path = Path(".")) -> dict[str, str]:
    """SHA-256 of every file the protocol freezes: config, cases, protocol and all code."""
    paths = FIXED + sorted(str(p.relative_to(root))
                           for p in Path(root, "src/dsoi").rglob("*.py"))
    return {p: hashlib.sha256(Path(root, p).read_bytes()).hexdigest() for p in paths}


def manifest_hash(files: dict[str, str]) -> str:
    return hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()


def deviations(frozen: dict[str, str], now: dict[str, str]) -> list[str]:
    return sorted(p for p in set(frozen) | set(now) if frozen.get(p) != now.get(p))


def _config() -> dict:
    return yaml.safe_load(CONFIG.read_text())


def _cases(cfg: dict) -> list:
    if not CASES.exists():
        save(generate(cfg["cases_per_category"], cfg["seed"], cfg["calibration_share"]), CASES)
    return load(CASES)


def _manifest() -> dict | None:
    return json.loads(MANIFEST.read_text()) if MANIFEST.exists() else None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("command", choices=["generate", "freeze", "dryrun", "run", "run-claude",
                                        "score", "status"])
    ap.add_argument("--yes", action="store_true", help="confirm a command that spends money")
    ap.add_argument("--pilot", action="store_true", help="run-claude: six cases, once")
    ap.add_argument("--reason", help="why a frozen protocol is frozen again")
    ap.add_argument("--models", nargs="*", help="a subset of the configured local models")
    args = ap.parse_args(argv)
    cfg = _config()
    cases = _cases(cfg)

    if args.command == "generate":
        print(f"{len(cases)} cases in {CASES}")
    elif args.command == "freeze":
        old = _manifest()
        if old and not args.reason:
            sys.exit("already frozen; a new freeze needs --reason (it is recorded)")
        files = frozen_files()
        history = (old or {}).get("history", []) + (
            [{"replaced": old["hash"], "at": old["frozen_at"], "reason": args.reason}]
            if old else [])
        MANIFEST.write_text(json.dumps(
            {"hash": manifest_hash(files), "frozen_at": datetime.now(UTC).isoformat(),
             "files": files, "history": history}, indent=1, sort_keys=True) + "\n")
        print(f"frozen {manifest_hash(files)[:16]} ({len(files)} files)")
    elif args.command == "dryrun":
        out = Path("results/e1v2-dryrun")
        if (out / "replies.jsonl").exists():
            (out / "replies.jsonl").unlink()
        run(cases, MockBackend(), ["mock"], cfg["repeats"], out / "replies.jsonl", "dryrun")
        s = score(cases, load_replies(out / "replies.jsonl"), cfg["alpha"], out, "dryrun")
        print(json.dumps(s["models"]["mock"]["stated"], indent=1))
    elif args.command == "run":
        m = _manifest()
        if m is None:
            sys.exit("not frozen: run `dsoi-e1v2 freeze` first")
        changed = deviations(m["files"], frozen_files())
        if changed:
            sys.exit("frozen files changed since the freeze: " + ", ".join(changed))
        models = args.models or cfg["local_models"]
        n = run(cases, OllamaBackend(cfg["temperature"]), models, cfg["repeats"], REPLIES,
                m["hash"], progress=lambda s: print(s, flush=True), workers=cfg["workers"])
        print(f"requested {n} replies")
    elif args.command == "run-claude":
        if not args.yes:
            sys.exit("run-claude spends money (budget in configs/e1v2.yaml): add --yes")
        m = _manifest()
        if m is None or deviations(m["files"], frozen_files()):
            sys.exit("not frozen, or frozen files changed: freeze (with --reason) first")
        import anthropic

        from dsoi.experiments.e1_harness import load_env
        load_env()
        c = cfg["claude"]
        backend = ClaudeBatch({k: dict(v) for k, v in c["prices_usd_per_mtok"].items()},
                              c["effort"], c["max_tokens"], c["cache_ttl"])
        picked, repeats, out = cases, cfg["repeats"], RESULTS
        if args.pilot:
            want = ["hedged_pct", "conflict", "two_entities", "scope_relative", "other_entity",
                    "injection"]
            picked = [next(x for x in cases if x.category == k and x.split == "calibration")
                      for k in want]
            repeats, out = 1, Path("results/e1v2-claude-pilot")
        spent = run_claude(picked, backend, args.models or c["models"], repeats,
                           out / "replies.jsonl", m["hash"], c["budget_usd"],
                           RESULTS / "claude-ledger.json", anthropic.Anthropic(),
                           progress=lambda s: print(s, flush=True))
        print(f"spent ${spent:.4f}")
    elif args.command == "score":
        m = _manifest() or {"hash": "not frozen", "files": {}}
        replies = load_replies(REPLIES)
        s = score(cases, replies, cfg["alpha"], RESULTS, m["hash"])
        s["replies_requested_under"] = sorted({r["manifest"] for r in replies.values()})
        s["deviations_since_freeze"] = deviations(m["files"], frozen_files())
        (RESULTS / "summary.json").write_text(json.dumps(s, indent=1, sort_keys=True) + "\n")
        print(json.dumps({k: v["coverage_stated"] for k, v in s["models"].items()}, indent=1))
    elif args.command == "status":
        m = _manifest()
        replies = load_replies(REPLIES)
        per = {}
        for model, *_ in replies:
            per[model] = per.get(model, 0) + 1
        print(json.dumps({"frozen": m["hash"][:16] if m else None,
                          "deviations": deviations(m["files"], frozen_files()) if m else None,
                          "replies": per}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
