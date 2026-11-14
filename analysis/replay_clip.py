"""Render a clip of a finished trial by replaying its transcript's API calls through a fresh sim.

    .venv/bin/python -m analysis.replay_clip results/transcripts/live_astra-<hash>.json [...]

The sim is seeded from the transcript's params, so replaying the same calls in the same order
reproduces the trial; the script checks the final TCP against the transcript and reports the gap.
Clips go to results/clips/<transcript stem>.mp4 (side camera, 15 fps, at most 10 s).
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

from sim.mj_bowl_sim import BowlSim
from sim.mj_sim import ApiCapReached, CollisionStop, PegSim
from sim.mj_sort_sim import SortSim
from sim.mj_trial import ROOT, _write_clip


def replay(path: Path) -> str:
    t = json.loads(path.read_text())
    params = t["params"]
    outcome = t.get("outcome") or {}
    bowl = path.name.startswith("bowl-") or outcome.get("colour") in ("cream", "grey", "black")
    sort = path.name.startswith("sort-") or "colour" in outcome
    sim = (BowlSim if bowl else SortSim if sort else PegSim)(params, record_every=20)
    try:
        for s in t["steps"]:
            args = s["args"]
            if s["api"] == "move_to":
                sim.move_to(args["xyz"], args["yaw"])
            elif s["api"] == "step_down":
                sim.step_down(args["mm"])
            elif s["api"] == "suction":
                sim.suction(args["on"])
            elif s["api"] == "bin_pose":
                sim.bin_pose(args["colour"])
            elif s["api"] == "bowl_pose":
                sim.bowl_pose(args["colour"])
            else:
                getattr(sim, s["api"])()
    except (ApiCapReached, CollisionStop):
        pass
    sim._capture()
    clip = _write_clip(sim.frames, path.stem)
    sim.close_renderer()
    last = t["steps"][-1]["tcp"]
    tip = sim.data.site_xpos[sim.tip]
    gap = math.dist(last, [float(v) for v in tip]) * 1000
    return f"{path.name}: {len(t['steps'])} calls replayed, final TCP gap {gap:.2f} mm, clip {clip}"


def main() -> None:
    for arg in sys.argv[1:]:
        print(replay(ROOT / arg if not Path(arg).is_absolute() else Path(arg)))


if __name__ == "__main__":
    main()
