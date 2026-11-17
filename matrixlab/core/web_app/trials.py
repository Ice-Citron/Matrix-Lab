"""One tidy frame of every trial in results/ for the Compare page, plus the per-trial detail readers.

Policy names follow results/<sim>_<policy>_<tag>.csv; results/local.csv and results/deploy.csv are
the astra-script peg rows. Rates use pass-1 rows and drop api_error rows, as analysis.coverage does.
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd

from matrixlab.evaluator.analysis.coverage import _drop_infra, _success, first_pass

ROOT = Path(__file__).resolve().parents[3]
RESULTS = ROOT / "results"
TASK = {"mujoco": "peg-in-hole", "sort": "bin-sorting", "bowl": "bowl-sorting", "ditto": "sphere-in-basket",
        "isaac": "orange-in-bowl"}
PARAMS = {"dx": "dx (m)", "dy": "dy (m)", "yaw": "yaw (deg)", "r": "distance from nominal (m)",
          "pose_error_mm": "camera error on the first estimate (mm)",
          "bx": "basket dx (m)", "by": "basket dy (m)"}
PLACEMENT = ("dx", "dy", "yaw", "bx", "by")  # the sampled placement parameters
TAG_RE = re.compile(r"_(cmp_smoke|cmp_pilot|dr|(?:local|deploy)(?:_v\d+\w*)?(?:_fresh|_invalid)?)$")
NOMINAL = {"mujoco": (0.45, 0.0), "sort": (0.45, -0.05), "bowl": (0.45, -0.05)}


def _parse(path: Path) -> tuple[str, str, str] | None:
    """(sim, policy, tag) for a results csv, or None for files the page does not show."""
    stem = path.stem
    if stem in ("local", "deploy"):
        return "mujoco", "astra-script", stem
    fresh = re.fullmatch(r"(local|deploy)_v(\d)_fresh", stem)  # the astra improve-loop sweeps
    if fresh:
        return "mujoco", f"astra-script-v{fresh.group(2)}", fresh.group(1)
    m = TAG_RE.search(stem)
    if not m:
        return None
    head, tag = stem[:m.start()], m.group(1)
    sim, _, policy = head.partition("_")
    if sim not in TASK or not policy:
        return None
    version = re.fullmatch(r"(local|deploy)_(v\d\w*)", tag)  # e.g. fable-script deploy_v2b: the v2 script
    if version:
        policy, tag = f"{policy}-{version.group(2).rstrip('b')}", version.group(1)
    return sim, policy, tag


def load_trials() -> pd.DataFrame:
    frames = []
    for path in sorted(RESULTS.glob("*.csv")):
        parsed = _parse(path)
        if not parsed:
            continue
        sim, policy, tag = parsed
        df = pd.read_csv(path)
        if "dx" not in df or not len(df):
            continue
        df = _drop_infra(first_pass(df)).copy()
        df["success"] = _success(df).astype(bool)
        df["failure_mode"] = df["failure_mode"].fillna("").astype(str)
        df["outcome"] = np.where(df["success"], "success", df["failure_mode"].replace("", "failure"))
        df["sim"], df["task"], df["policy"], df["tag"] = sim, TASK[sim], policy, tag
        df["set"] = ("local" if tag.startswith("local") else "deploy" if tag.startswith("deploy")
                     else tag[4:] if tag.startswith("cmp_") else tag)
        df["r"] = np.hypot(df["dx"].astype(float), df["dy"].astype(float))
        df["csv"] = str(path.relative_to(ROOT))
        frames.append(df)
    if not frames:
        return pd.DataFrame(columns=["sim", "task", "policy", "tag", "set", "dx", "dy", "yaw", "seed", "success",
                                     "outcome", "r", "pose_error_mm", "clip", "transcript", "csv"])
    return pd.concat(frames, ignore_index=True)


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, centre - half), min(1.0, centre + half))


def binned_rate(df: pd.DataFrame, param: str, bins: int, lo: float | None = None, hi: float | None = None) -> pd.DataFrame:
    """Success rate per policy per bin of `param` with the Wilson 95 % interval."""
    if df.empty:
        return pd.DataFrame(columns=["policy", "x", "x_lo", "x_hi", "n", "k", "rate", "lo", "hi"])
    v = df[param].astype(float)
    lo = v.min() if lo is None else lo
    hi = v.max() if hi is None else hi
    edges = np.linspace(lo, hi, bins + 1)
    idx = np.clip(np.digitize(v, edges) - 1, 0, bins - 1)
    rows = []
    for policy, g in df.assign(_bin=idx).groupby("policy"):
        for b, gb in g.groupby("_bin"):
            k, n = int(gb["success"].sum()), len(gb)
            w = wilson(k, n)
            rows.append({"policy": policy, "x": float((edges[b] + edges[b + 1]) / 2), "x_lo": float(edges[b]),
                         "x_hi": float(edges[b + 1]), "n": n, "k": k, "rate": k / n, "lo": w[0], "hi": w[1]})
    return pd.DataFrame(rows)


def read_transcript(rel: str) -> dict | None:
    p = ROOT / rel
    if not p.is_file():
        return None
    try:
        t = json.loads(p.read_text())
    except ValueError:
        return None
    return t if isinstance(t, dict) else None


def is_mujoco(t: dict) -> bool:
    steps = t.get("steps")
    return isinstance(steps, list) and bool(steps) and isinstance(steps[0], dict) and "tcp" in steps[0]


def trajectory(t: dict) -> pd.DataFrame:
    """TCP after every API call of a MuJoCo transcript."""
    if not is_mujoco(t):
        return pd.DataFrame()
    rows = [{"i": s["i"], "api": s["api"], "x": s["tcp"][0], "y": s["tcp"][1], "z": s["tcp"][2],
             "force_N": s.get("force_N")} for s in t["steps"] if s.get("tcp")]
    return pd.DataFrame(rows)


def ditto_trajectory(transcript_rel: str) -> pd.DataFrame:
    """End-effector position per control step from the Ditto run folder that produced the transcript."""
    stem = Path(transcript_rel).stem
    h = stem.rsplit("-", 1)[-1]
    files = sorted((ROOT / "results" / "runs").glob(f"{h}-*/evaluation/*/evals/*/trajectory-steps.jsonl"))
    if not files:
        return pd.DataFrame()
    rows = []
    with open(files[-1]) as fh:
        for line in fh:
            try:
                d = json.loads(line)
            except ValueError:
                continue
            if d.get("eef_pos"):
                rows.append({"i": d.get("step"), "api": "control step", "x": d["eef_pos"][0], "y": d["eef_pos"][1],
                             "z": d["eef_pos"][2], "force_N": None})
    return pd.DataFrame(rows)


def reason(t: dict) -> str:
    """Why the trial ended, from the outcome and the agent log when there is one."""
    if "oracle_success" in t:  # a Ditto transcript: the framework's result record
        meta = t.get("policy_metadata") or {}
        parts = [f"{'success' if t.get('oracle_success') else 'failure'} by the task oracle",
                 f"run status {t.get('status')} ended by {t.get('termination_reason')}",
                 f"{t.get('steps')} control steps over {t.get('sim_time_s')} s of sim time",
                 f"progress {t.get('final_progress_score')} at the end (peak {t.get('progress_score')})"]
        if meta.get("policy"):
            parts.append(f"policy {meta['policy']}")
        return " · ".join(str(x) for x in parts)
    o = t.get("outcome") or {}
    parts = [f"{'success' if o.get('success') else o.get('failure_mode') or 'failure'} after {o.get('api_calls')} API calls"]
    for k, unit in (("depth_mm", "mm deep"), ("lateral_mm", "mm lateral"), ("yaw_err_deg", "deg yaw error"),
                    ("max_force_N", "N peak force"), ("pose_error_mm", "mm camera error")):
        if o.get(k) is not None:
            parts.append(f"{o[k]:.1f} {unit}")
    if o.get("error"):
        parts.append(f"error: {o['error']}")
    a = t.get("agent")
    if a:
        parts.append(f"{a.get('mode')} mode ended by {a.get('end') or 'the sim'} after {a.get('model_calls')} model calls")
        errs = [c.get("script_error") or c.get("parse_error") for c in a.get("calls", []) if c.get("script_error") or c.get("parse_error")]
        if errs:
            parts.append("model-side errors: " + "; ".join(e[:120] for e in errs))
    return " · ".join(parts)


def last_reply(t: dict, limit: int = 1200) -> str | None:
    a = t.get("agent")
    if not a or not a.get("calls"):
        return None
    r = a["calls"][-1].get("reply") or ""
    return r if len(r) <= limit else r[:limit] + f"\n… ({len(r) - limit} more characters)"


def clip_path(transcript_rel: str, render: bool = True, row_clip: str | None = None) -> Path | None:
    """The csv row's clip when it exists, else results/clips/<transcript stem>.mp4, rendered by
    analysis.replay_clip when missing and the transcript is a MuJoCo one."""
    if isinstance(row_clip, str) and (ROOT / row_clip).is_file():
        return ROOT / row_clip
    stem = Path(transcript_rel).stem
    clip = RESULTS / "clips" / f"{stem}.mp4"
    if clip.is_file():
        return clip
    if not render or not is_mujoco(read_transcript(transcript_rel) or {}):
        return None
    from matrixlab.core.web_app.replay_clip import replay
    try:
        replay(ROOT / transcript_rel)
    except Exception:  # noqa: BLE001 - a trial without a clip is still worth showing
        return None
    return clip if clip.is_file() else None


def mcap_trajectory(path: str, topic: str = "/aic_controller/controller_state", every: int = 10) -> pd.DataFrame:
    """TCP position over time from a ROS 2 mcap bag (every n-th message)."""
    from mcap_ros2.reader import read_ros2_messages
    rows, t0 = [], None
    for i, m in enumerate(read_ros2_messages(path, topics=[topic])):
        if i % every:
            continue
        t0 = m.log_time_ns if t0 is None else t0
        p = m.ros_msg.tcp_pose.position
        rows.append({"t_s": (m.log_time_ns - t0) / 1e9, "x": p.x, "y": p.y, "z": p.z})
    return pd.DataFrame(rows)
