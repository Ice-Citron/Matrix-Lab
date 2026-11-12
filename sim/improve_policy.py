"""Close the loop: show Astra its own policy's sweep failures and ask for a better version.

  .venv/bin/python -m sim.improve_policy v1 v2                       # reads results/deploy.csv
  .venv/bin/python -m sim.improve_policy v2 v3 --sweep results/deploy_v2.csv
  GAPHUNTER_SIM=mujoco GAPHUNTER_POLICY=v2 .venv/bin/python -m runner.batch --box deploy --n 600 \
      --tag deploy_v2 --workers 8                                      # then --box local --n 30 --tag local_v2

Reads the sweep CSV, bins the pass-1 failures into 4 cm cells of (dx, dy), summarises the worst eight
from their transcripts, and sends the task, the current module and those summaries to the same model
that wrote the module. The reply is saved like sim.write_policy does: results/policies/<dst>.py and
<dst>.exchange.json (plus the source policy and sweep it was improved from).
"""
from __future__ import annotations

import argparse
import inspect
import json
import re

import pandas as pd

from sim.mj_sim import NOMINAL_HOLE
from sim.write_policy import MODEL, ROOT, TASK, ask_astra, ask_opus


#RUCHA'S CODE REVIEW

#PURPOSE: this script is supposed to take the results of a policy sweep, identify the worst regions of failure , and then ask the model to improve the policy based on those failures. It reads the sweep CSV, bins the failures into cells, summarizes the worst ones, and sends this information to the model for improvement
# The improved policy is then saved as a new version
#DEPENDS ON: sim.mj_sim, sim.write_policy, pandas, argparse, inspect, json, re
# USED BY: sim.improve_policy, sim.mj_trial, sim.mj_sort_trial, sim.write_policy
# there are no todos for this (?) at the moment 



CELL_M = 0.04
N_CELLS = 8
LAST_CALLS = 6


def worst_cells(df: pd.DataFrame) -> list[pd.Series]:
    """One failed trial per cell for the N_CELLS cells with the most failures (largest pose error picked)."""
    rows = df[df["pass"] == 1].copy()
    rows["cell"] = list(zip((rows["dx"] // CELL_M).astype(int), (rows["dy"] // CELL_M).astype(int)))
    fails = rows[~rows["success"]]
    cells = fails.groupby("cell").size().sort_values(ascending=False).index[:N_CELLS]
    return [fails[fails["cell"] == c].sort_values("pose_error_mm").iloc[-1] for c in cells]


def summarise(row: pd.Series) -> str:
    t = json.loads((ROOT / row["transcript"]).read_text())
    est = next(s["result"] for s in t["steps"] if s["api"] == "hole_pose")
    true_x, true_y = NOMINAL_HOLE + (row["dx"], row["dy"])
    moves = [s for s in t["steps"] if s["api"] in ("move_to", "step_down")][-LAST_CALLS:]
    out = t["outcome"]
    lines = [
        f"- true hole x={true_x:.3f} y={true_y:.3f} yaw={row['yaw']:.1f} deg; camera estimate "
        f"x={est[0][0]:.3f} y={est[0][1]:.3f} yaw={est[1]:.1f} deg; estimate error {row['pose_error_mm']:.1f} mm",
        f"  outcome: {out['failure_mode']} after {out['api_calls']} calls, depth {out['depth_mm']:.1f} mm, "
        f"lateral {out['lateral_mm']:.1f} mm, peak force {out['max_force_N']:.1f} N",
        "  last calls: " + "; ".join(
            f"{s['api']}({json.dumps(s['args'])}) -> tcp {s['tcp']} force {s['force_N']}" for s in moves),
    ]
    return "\n".join(lines)


def build_prompt(code: str, df: pd.DataFrame, summaries: list[str]) -> str:
    from sim import api
    done = df[df["pass"] == 1]
    rate = 100 * done["success"].mean()
    return (TASK.replace("{api_source}", inspect.getsource(api))
            + "\n\nThis is the current policy module:\n\n```python\n" + code + "```\n\n"
            + f"This policy was tested at {len(done)} placements across x = 0.45 +/- 0.20 m, y = +/- 0.20 m, "
            f"yaw +/- 90 degrees. It succeeded in {rate:.0f}% of trials. "
            "Here are the worst failure regions and what happened in each:\n\n" + "\n".join(summaries)
            + "\n\nRewrite the module so it succeeds in these regions too, keeping the same API and the "
            "300-call budget. Return the complete module.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("src", help="policy version the sweep was run with, e.g. v1")
    ap.add_argument("dst", help="version to write, e.g. v2")
    ap.add_argument("--sweep", default="results/deploy.csv")
    a = ap.parse_args()
    policies = ROOT / "results" / "policies"
    code = (policies / f"{a.src}.py").read_text()
    df = pd.read_csv(ROOT / a.sweep)
    prompt = build_prompt(code, df, [summarise(r) for r in worst_cells(df)])
    ask = ask_opus if MODEL.startswith("claude") else ask_astra
    text, meta = ask(prompt)
    match = re.search(r"```python\n(.*?)```", text, re.S)
    if not match:
        raise SystemExit(f"No code block in the model's reply:\n{text}")
    (policies / f"{a.dst}.py").write_text(match.group(1))
    (policies / f"{a.dst}.exchange.json").write_text(json.dumps(
        {"model": meta["model"], "usage": meta["usage"], "prompt": prompt, "response": text,
         "code": match.group(1), "policy_path": f"results/policies/{a.dst}.py",
         "source_policy": f"results/policies/{a.src}.py", "sweep": a.sweep}, indent=1))
    print(text)


if __name__ == "__main__":
    main()
