"""Stop a model-in-the-loop batch when too many trials run away.

    .venv/bin/python -m runner.watchdog --policies live_fable fly_fable --limit 5 --max-output 40000

Every INTERVAL seconds it reads the transcripts written since it started, keeps the ones that carry
an "agent" log for one of the named policies, and counts the runaway ones: a wall-clock timeout, or
more than --max-output output tokens over the trial. Past --limit it kills every runner.batch process
whose command line names one of the policies, prints what it counted, and exits 3.
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRANSCRIPTS = ROOT / "results" / "transcripts"
INTERVAL = 30


def runaway(path: Path, max_output: int) -> str | None:
    try:
        t = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    if not isinstance(t, dict):
        return None
    agent = t.get("agent")
    if not agent:
        return None
    out = sum(((c.get("usage") or {}).get("output_tokens") or 0) for c in agent.get("calls", []))
    error = str((t.get("outcome") or {}).get("error") or "")
    if error.startswith("wall clock"):
        return f"wall-clock timeout after {out} output tokens"
    if out > max_output:
        return f"{out} output tokens"
    return None


def policy_of(path: Path, t: dict) -> str:
    return t.get("policy") or path.name.split("-", 1)[0]


def batch_pids(policies: list[str]) -> list[int]:
    out = subprocess.run(["pgrep", "-af", "runner.batch"], capture_output=True, text=True).stdout
    pids = []
    for line in out.splitlines():
        pid, _, cmd = line.partition(" ")
        if any(f"--policy {p}" in cmd for p in policies) and "watchdog" not in cmd:
            pids.append(int(pid))
    return pids


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--policies", nargs="+", required=True)
    ap.add_argument("--limit", type=int, default=5, help="stop when more than this many trials ran away")
    ap.add_argument("--max-output", type=int, default=40000)
    ap.add_argument("--once", action="store_true", help="count every transcript on disk once and exit")
    a = ap.parse_args()
    started = 0.0 if a.once else time.time()
    while True:
        seen = {}
        for path in TRANSCRIPTS.glob("*.json"):
            if path.stat().st_mtime < started:
                continue
            why = runaway(path, a.max_output)
            if why:
                t = json.loads(path.read_text())
                if policy_of(path, t) in a.policies:
                    seen[path.name] = why
        if a.once:
            for k, v in seen.items():
                print(k, v)
            print(f"{len(seen)} runaway trials")
            return
        if len(seen) > a.limit:
            pids = batch_pids(a.policies)
            for pid in pids:
                os.kill(pid, signal.SIGTERM)
            print(f"watchdog: {len(seen)} runaway trials (> {a.limit}); killed batch pids {pids}", file=sys.stderr)
            for k, v in seen.items():
                print(f"  {k}: {v}", file=sys.stderr)
            sys.exit(3)
        if not batch_pids(a.policies):
            print(f"watchdog: no batch running for {a.policies}; {len(seen)} runaway trials; exiting")
            return
        time.sleep(INTERVAL)


if __name__ == "__main__":
    main()
