"""One tidy frame of every trial for the Compare page, from the flat results/<sim>_<policy>_<tag>.csv files
that bootstrap.py writes from the dataset."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

from bootstrap import RESULTS, build

TASK = {"mujoco": "peg-in-hole", "sort": "bin-sorting", "bowl": "bowl-sorting", "ditto": "sphere-in-basket",
        "isaac": "orange-in-bowl"}
PARAMS = {"dx": "dx (m)", "dy": "dy (m)", "yaw": "yaw (deg)", "r": "distance from nominal (m)",
          "pose_error_mm": "camera error on the first estimate (mm)", "bx": "basket dx (m)", "by": "basket dy (m)"}
PLACEMENT = ("dx", "dy", "yaw", "bx", "by")


def _success(df):
    s = df["success"]
    if s.dtype == object:
        s = s.astype(str).str.lower().isin(("true", "1", "yes"))
    return s.astype(float)


def load_trials() -> pd.DataFrame:
    build()
    frames = []
    for path in sorted(RESULTS.glob("*.csv")):
        sim, policy, tag = path.stem.split("_", 2)
        if sim not in TASK:
            continue
        df = pd.read_csv(path)
        if "dx" not in df or not len(df):
            continue
        if "pass" in df:
            df = df[df["pass"].fillna(1) == 1]
        if "failure_mode" in df:
            df = df[df["failure_mode"].fillna("").astype(str) != "api_error"]
        df = df.copy()
        df["success"] = _success(df).astype(bool)
        df["failure_mode"] = df.get("failure_mode", pd.Series("", index=df.index)).fillna("").astype(str)
        df["outcome"] = np.where(df["success"], "success", df["failure_mode"].replace("", "failure"))
        df["sim"], df["task"], df["policy"], df["tag"] = sim, TASK[sim], policy, tag
        df["set"] = ("local" if tag.startswith("local") else "deploy" if tag.startswith("deploy")
                     else tag[4:] if tag.startswith("cmp_") else tag)
        df["r"] = np.hypot(df["dx"].astype(float), df["dy"].astype(float))
        df["csv"] = path.name
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, centre - half), min(1.0, centre + half))


def binned_rate(df: pd.DataFrame, param: str, bins: int) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["policy", "x", "n", "k", "rate", "lo", "hi"])
    v = df[param].astype(float)
    edges = np.linspace(v.min(), v.max(), bins + 1)
    idx = np.clip(np.digitize(v, edges) - 1, 0, bins - 1)
    rows = []
    for policy, g in df.assign(_bin=idx).groupby("policy"):
        for b, gb in g.groupby("_bin"):
            k, n = int(gb["success"].sum()), len(gb)
            w = wilson(k, n)
            rows.append({"policy": policy, "x": float((edges[b] + edges[b + 1]) / 2), "n": n, "k": k,
                         "rate": k / n, "lo": w[0], "hi": w[1]})
    return pd.DataFrame(rows)


def reason(row) -> str:
    parts = [f"{'success' if row['success'] else row['failure_mode'] or 'failure'}"]
    if pd.notna(row.get("n_decisions")):
        parts.append(f"{int(row['n_decisions'])} API calls or decisions")
    if pd.notna(row.get("pose_error_mm")):
        parts.append(f"{row['pose_error_mm']:.1f} mm camera error on the first estimate")
    if pd.notna(row.get("t_wall")):
        parts.append(f"{row['t_wall']:.0f} s wall time")
    return " · ".join(parts)
