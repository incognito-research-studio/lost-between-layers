"""dsoi-e1v3: the phase 3 prospective study (data/e1v3/PREREGISTRATION.md).

  generate     write the case set to data/e1v3/cases.jsonl (needs seed_token in the config)
  plan         requests per model and job, with the expected and worst-case cost
  freeze       hash protocol, config, cases and code into data/e1v3/manifest.json
  run-claude   Claude models through the Message Batches API (spends money: needs --yes);
               --pilot sends a small fixed set of pilot cases to measure tokens, outside the
               study's cases and store
  run          local models through Ollama (no cost; heavy local load)
  status       stored replies per model, kind and arm, and whether the freeze still holds
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import yaml

from dsoi.experiments.e1v2_cli import deviations, manifest_hash
from dsoi.experiments.e1v3_cases import generate, load, save, seed_from_token
from dsoi.experiments.e1v3_run import (
    ClaudeV3,
    Job,
    OllamaV3,
    load_replies,
    run_claude,
    run_ollama,
)
from dsoi.experiments.e1v3_variants import variants

CONFIG = Path("configs/e1v3.yaml")
DATA = Path("data/e1v3")
CASES = DATA / "cases.jsonl"
MANIFEST = DATA / "manifest.json"
RESULTS = Path("results/e1v3")
REPLIES = RESULTS / "replies.jsonl"
FIXED = ["configs/e1v3.yaml", "configs/versions.yaml", "data/e1v3/PREREGISTRATION.md",
         "data/e1v3/TRUTH_RULES.md", "data/e1v3/cases.jsonl"]
PILOT_CATEGORIES = ("hedged_pct", "conflict", "scope_relative", "resent", "amend_size",
                    "cancel")


def frozen_files(root: Path = Path(".")) -> dict[str, str]:
    import hashlib
    paths = FIXED + sorted(str(p.relative_to(root))
                           for p in Path(root, "src/dsoi").rglob("*.py"))
    return {p: hashlib.sha256(Path(root, p).read_bytes()).hexdigest() for p in paths}


def _config() -> dict:
    return yaml.safe_load(CONFIG.read_text())


def _manifest() -> dict | None:
    return json.loads(MANIFEST.read_text()) if MANIFEST.exists() else None


def replicate_subset(cases: list, share: float, seed: int) -> tuple:
    """A seeded share of the cases, for a second reply from single-repeat models."""
    rng = np.random.default_rng(seed + 2)
    k = round(share * len(cases))
    return tuple(cases[i] for i in sorted(rng.permutation(len(cases))[:k]))


def jobs(cases: list, cfg: dict, section: dict, seed: int) -> list[Job]:
    """The study's jobs for one backend section (claude or local)."""
    test = tuple(c for c in cases if c.split == "test")
    var = tuple(variants(list(cases)))
    arms = tuple(cfg["arms"])
    rep = section["repeats"]
    if isinstance(rep, int):                       # local: one number for every model
        rep = dict.fromkeys(section["models"], rep)
    single = {m: 2 for m, n in rep.items() if n == 1}
    base = section.get("baseline_repeats", rep)
    var_rep = section.get("variant_repeats", dict.fromkeys(rep, 1))
    out = [Job("interp", tuple(cases), arms, rep),
           Job("decide", test, ("decide",), base),
           Job("interp", var, ("ref",), var_rep),
           Job("decide", var, ("decide",), var_rep)]
    if single:
        out.append(Job("interp", replicate_subset(cases, cfg["replicate_share"], seed),
                       arms, single))
    return out


def count(job_list: list[Job], model: str) -> int:
    """Distinct requests a model needs: a case with k messages needs k interp requests per
    arm and repeat; a request two jobs share (the replicate job's first repeat) counts once."""
    keys = set()
    for j in job_list:
        for r in range(1, j.repeats.get(model, 0) + 1):
            for c in j.cases:
                if j.kind == "decide":
                    keys.add(("decide", c.id, r))
                else:
                    keys |= {("interp", a, c.id, r, i) for a in j.arms
                             for i in range(len(c.messages))}
    return len(keys)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("command", choices=["generate", "plan", "freeze", "run-claude", "run",
                                        "status"])
    ap.add_argument("--yes", action="store_true", help="confirm a command that spends money")
    ap.add_argument("--pilot", action="store_true", help="run-claude: pilot cases only")
    ap.add_argument("--reason", help="why a frozen protocol is frozen again")
    ap.add_argument("--models", nargs="*", help="a subset of the configured models")
    ap.add_argument("--output-tokens", type=float, nargs=2, default=(300, 1500),
                    help="plan: low and high output tokens per request (with thinking)")
    args = ap.parse_args(argv)
    cfg = _config()

    if args.command == "generate":
        if CASES.exists():
            sys.exit(f"{CASES} exists; the case set is generated once")
        seed = seed_from_token(str(cfg["seed_token"]))
        save(generate(cfg["calibration_per_category"], cfg["test_per_category"], seed), CASES)
        print(f"{len(load(CASES))} cases in {CASES} (seed {seed})")
        return 0

    if args.command == "plan":
        token = str(cfg["seed_token"])
        seed = seed_from_token(token if token.upper() != "DECIDE" else "plan-preview")
        cases = (load(CASES) if CASES.exists() else
                 generate(cfg["calibration_per_category"], cfg["test_per_category"], seed))
        c = cfg["claude"]
        backend = ClaudeV3({k: dict(v) for k, v in c["prices_usd_per_mtok"].items()},
                           c["effort"], c["max_tokens"], c["cache_ttl"])
        job_list = jobs(cases, cfg, c, seed)
        lo_t, hi_t = args.output_tokens
        rows, total = {}, [0.0, 0.0, 0.0]
        for m in args.models or c["models"]:
            n = count(job_list, m)
            lo = n * backend.cost(m, {"input_tokens": 1500, "output_tokens": lo_t})
            hi = n * backend.cost(m, {"input_tokens": 1500, "output_tokens": hi_t})
            worst = backend.worst_case(m, n)
            rows[m] = {"requests": n, "expected_usd": [round(lo, 2), round(hi, 2)],
                       "worst_case_usd": round(worst, 2)}
            total = [total[0] + lo, total[1] + hi, total[2] + worst]
        print(json.dumps({"cases": len(cases), "variants": len(variants(list(cases))),
                          "models": rows,
                          "total_expected_usd": [round(total[0], 2), round(total[1], 2)],
                          "total_worst_case_usd": round(total[2], 2),
                          "budget_usd": c["budget_usd"]}, indent=1))
        return 0

    if args.command == "freeze":
        if not CASES.exists():
            sys.exit("generate the cases first (after the seed token is fixed)")
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
        return 0

    if args.command == "status":
        m = _manifest()
        last, _ = load_replies(REPLIES)
        per: dict = {}
        for kind, model, arm, *_ in last:
            per.setdefault(model, {}).setdefault(f"{kind}/{arm}", 0)
            per[model][f"{kind}/{arm}"] += 1
        print(json.dumps({"frozen": m["hash"][:16] if m else None,
                          "deviations": deviations(m["files"], frozen_files()) if m else None,
                          "replies": per}, indent=1))
        return 0

    if args.command == "run-claude":
        if not args.yes:
            sys.exit("run-claude spends money (budget in configs/e1v3.yaml): add --yes")
        c = cfg["claude"]
        backend = ClaudeV3({k: dict(v) for k, v in c["prices_usd_per_mtok"].items()},
                           c["effort"], c["max_tokens"], c["cache_ttl"])
        models = args.models or c["models"]
        if args.pilot:
            pcases = generate(1, 0, seed_from_token("pilot"))
            pcases = [x for x in pcases if x.category in PILOT_CATEGORIES]
            job_list = [Job("interp", tuple(pcases), tuple(cfg["arms"]), dict.fromkeys(
                models, 1)), Job("decide", tuple(pcases[:2]), ("decide",),
                                 dict.fromkeys(models, 1))]
            store, ledger, manifest = (Path("results/e1v3-pilot/replies.jsonl"),
                                       Path("results/e1v3-pilot/ledger.json"), "pilot")
        else:
            m = _manifest()
            if m is None or deviations(m["files"], frozen_files()):
                sys.exit("not frozen, or frozen files changed: freeze (with --reason) first")
            job_list = jobs(load(CASES), cfg, c, seed_from_token(str(cfg["seed_token"])))
            store, ledger, manifest = REPLIES, RESULTS / "claude-ledger.json", m["hash"]
        import anthropic

        from dsoi.experiments.e1_harness import load_env
        load_env()
        spent = run_claude(job_list, backend, models, store, manifest, c["budget_usd"], ledger,
                           anthropic.Anthropic(), progress=lambda s: print(s, flush=True),
                           max_attempts=cfg["max_attempts"], batch_size=c["batch_size"])
        print(f"spent ${spent:.4f}")
        return 0

    if args.command == "run":
        m = _manifest()
        if m is None or deviations(m["files"], frozen_files()):
            sys.exit("not frozen, or frozen files changed: freeze (with --reason) first")
        loc = cfg["local"]
        models = args.models or loc["models"]
        job_list = jobs(load(CASES), cfg, {**loc, "models": models},
                        seed_from_token(str(cfg["seed_token"])))
        n = run_ollama(job_list, OllamaV3(loc["temperature"], max_tokens=loc["max_tokens"]),
                       models, REPLIES, m["hash"], progress=lambda s: print(s, flush=True),
                       max_attempts=cfg["max_attempts"])
        print(f"requested {n} replies")
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
