"""Lay the dataset's <task>/<policy>/trials/<tag>.csv files out as results/<sim>_<policy>_<tag>.csv
with the clip column pointing at the dataset's videos, which is the flat layout the page reads."""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
from huggingface_hub import HfApi, snapshot_download

DATASET = "Matrix-Lab/Matrix-Lab-Preliminary-Experiments"
URL = f"https://huggingface.co/datasets/{DATASET}/resolve/main"
SIM = {"peg-in-hole": "mujoco", "bin-sorting": "sort", "bowl-sorting": "bowl", "sphere-in-basket": "ditto",
       "so101-orange": "isaac"}
RESULTS = Path(__file__).resolve().parent / "results"


def build() -> Path:
    if (RESULTS / ".done").exists():
        return RESULTS
    root = Path(snapshot_download(DATASET, repo_type="dataset", allow_patterns=["*/*/trials/*.csv"]))
    videos = {f for f in HfApi().list_repo_files(DATASET, repo_type="dataset") if f.endswith(".mp4") and "/videos/" in f}
    RESULTS.mkdir(exist_ok=True)
    for csv in sorted(root.glob("*/*/trials/*.csv")):
        task, policy, _, name = csv.relative_to(root).parts
        if task not in SIM:
            continue
        df = pd.read_csv(csv)
        if "dx" not in df:
            continue
        clips = []
        for i, row in df.iterrows():
            stem = None
            for col in ("clip", "transcript"):
                v = row.get(col)
                if isinstance(v, str) and v:
                    stem = os.path.basename(v).rsplit(".", 1)[0]
                    break
            rel = f"{task}/{policy}/videos/{stem}.mp4" if stem else None
            clips.append(f"{URL}/{rel}" if rel in videos else None)
        df["clip"] = clips
        df.to_csv(RESULTS / f"{SIM[task]}_{policy}_{name}", index=False)
    (RESULTS / ".done").touch()
    return RESULTS


if __name__ == "__main__":
    print(build(), len(list(RESULTS.glob("*.csv"))), "csv files")
