"""Post-hoc per-run harness for the round-2 fault trial (13 Nov 2026).

    PYTHONPATH=<patched src> python tools/ft_perrun_harness.py --out runs.json \
        --episodes DIR [--hit-lines JSON]

Runs exactly the worlds of tools/ft_round2_harness.py, in the same order, through the frozen
`ft_harness.evaluate` with the extended trace, and adds one observation per run: which of
the patch's lines executed during that run (`hits`), recorded with sys.monitoring. Lines
that execute at import, before the first run, are recorded under the pseudo-run "import".
The checks, worlds and outcomes are the frozen ones; this file adds no check and changes no
classification. `--hit-lines` is a JSON object {path relative to src/: [line numbers]}.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

TOOL_ID = 4
_current = ["import"]
_hits: dict[str, set] = {}
_targets: dict[str, set[int]] = {}


_real: dict[str, str] = {}
_rel: dict[str, str] = {}  # real path -> path relative to src/, so no local path is stored


def _on_line(code, line):
    name = code.co_filename
    real = _real.get(name)
    if real is None:
        real = _real[name] = os.path.realpath(name)
    lines = _targets.get(real)
    if lines is None or line not in lines:
        return sys.monitoring.DISABLE
    _hits.setdefault(_current[0], set()).add(f"{_rel[real]}:{line}")
    return None


def install(hit_lines: dict[str, list[int]]) -> None:
    import dsoi
    root = Path(dsoi.__file__).resolve().parent.parent  # the patched src/
    for rel, lines in hit_lines.items():
        _targets[os.path.realpath(root / rel)] = set(lines)
        _rel[os.path.realpath(root / rel)] = rel
    sys.monitoring.use_tool_id(TOOL_ID, "ft_perrun")
    sys.monitoring.register_callback(TOOL_ID, sys.monitoring.events.LINE, _on_line)
    sys.monitoring.set_events(TOOL_ID, sys.monitoring.events.LINE)


def _plain(x):
    try:
        return json.loads(json.dumps(x, default=str))
    except Exception:
        return str(x)


def records_of(run) -> list:
    """The episode's audit records, read after the frozen evaluation (for adjudication only)."""
    out = []
    for r in getattr(run.p.audit, "records", []):
        out.append({k: _plain(getattr(r, k, None)) for k in
                    ("t", "outcome", "gate_outcome", "x_released", "x_exec", "state",
                     "admission", "gate_detail")})
    return out


def replays_of(run) -> list:
    """Replay outputs per decided record, as the extended trace computes them before hashing."""
    import ft_trace_ext
    p = run.p
    return [[r.id, ft_trace_ext._outcome(lambda r=r: p.replay(r.id)),
             ft_trace_ext._outcome(lambda r=r: p.replay(r.id, from_store=True))]
            for r in p.audit.decisions()]


def reload_of(run) -> dict:
    """Save and reload the log as the extended trace does; compare content, not counts."""
    import tempfile
    from dataclasses import asdict
    try:
        from dsoi.layers.l6_audit import AuditLog
        p = run.p
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "audit.jsonl"
            head = p.audit.to_jsonl(path)
            loaded = AuditLog.from_jsonl(path, expected_head=head)
        return {"ok": True,
                "same_deltas": [repr(x) for x in loaded.deltas] == [repr(x) for x in p.audit.deltas],
                "same_ingests": [asdict(x) for x in loaded.ingests] == [asdict(x) for x in p.audit.ingests],
                "same_records": [r.record_hash for r in loaded.records]
                == [r.record_hash for r in p.audit.records],
                "n_deltas": [len(p.audit.deltas), len(loaded.deltas)],
                "n_ingests": [len(p.audit.ingests), len(loaded.ingests)]}
    except Exception as exc:
        msg = str(exc).replace(str(Path(tempfile.gettempdir()).resolve()), "<tmp>")
        msg = msg.replace(tempfile.gettempdir(), "<tmp>")
        return {"ok": False, "raised": f"{type(exc).__name__}: {msg}"[:200]}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--episodes", type=Path, required=True)
    ap.add_argument("--hit-lines", type=Path)
    args = ap.parse_args(argv)
    if args.hit_lines:
        install(json.loads(args.hit_lines.read_text()))
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import ft_harness
    import ft_round2_harness
    import ft_trace_ext  # noqa: F401  (installs the extended trace, as in round 2)
    gens = [ft_harness.e1v2_runs(), ft_harness.scenario_runs(),
            ft_round2_harness.dir_runs(args.episodes, "episodes")]
    runs = {}
    for gen in gens:
        for run_id, go in gen:
            _current[0] = run_id
            built = []

            def make(go=go, built=built):
                run = go()
                built.append(run)
                return run

            runs[run_id] = ft_harness.evaluate(make)
            runs[run_id]["hits"] = sorted(_hits.get(run_id, ()))
            if built and not run_id.startswith("e1v2/"):
                runs[run_id]["records"] = {"records": records_of(built[0]),
                                           "replays": replays_of(built[0]),
                                           "reload": reload_of(built[0])}
            _current[0] = "between"  # code run while a world generator advances
    meta = {c.name: {"config": c.config, "layers": list(c.layers), "timing": c.timing,
                     "relative": c.relative} for c in ft_harness.CHECKS}
    args.out.write_text(json.dumps({"checks": meta, "runs": runs,
                                    "import_hits": sorted(_hits.get("import", ())),
                                    "between_hits": sorted(_hits.get("between", ()))},
                                   sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
