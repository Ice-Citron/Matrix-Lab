"""Build results/videos/<task>/<policy>/<set>_seed<seed>_<outcome>.mp4 as links to the clips.

    .venv/bin/python -m analysis.video_index

The clips themselves stay in results/clips under their trial hash, which the csv files and the app
reference. This tree is the human view of the same files: one folder per task, one per policy
(model and version), one link per trial that has a clip. It is rebuilt from scratch each run.
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path

from matrixlab.core.web_app.trials import ROOT, load_trials

VIDEOS = ROOT / "results" / "videos"


def main() -> None:
    if VIDEOS.exists():
        shutil.rmtree(VIDEOS)
    df = load_trials()
    made = 0
    for r in df.itertuples():
        clip = r.clip if isinstance(r.clip, str) else None
        if not clip:
            clip = f"results/clips/{Path(r.transcript).stem}.mp4" if isinstance(r.transcript, str) else None
        if not clip or not (ROOT / clip).is_file():
            continue
        folder = VIDEOS / r.task / r.policy
        folder.mkdir(parents=True, exist_ok=True)
        name = f"{r.set}_seed{int(r.seed)}_{r.outcome}.mp4"
        os.symlink(os.path.relpath(ROOT / clip, folder), folder / name)
        made += 1
    for task in sorted(p for p in VIDEOS.iterdir() if p.is_dir()):
        print(task.name, {p.name: len(list(p.iterdir())) for p in sorted(task.iterdir())})
    print(f"{made} links under {VIDEOS.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
