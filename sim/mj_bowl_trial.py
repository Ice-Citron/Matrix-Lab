"""run_trial for the MuJoCo cube-in-bowl scene: sim.mj_sort_trial with round bowls and sim.bowl_api."""
from __future__ import annotations

from sim import bowl_api
from sim.mj_bowl_sim import BowlSim
from sim.mj_sort_trial import run_sort_trial


def run_bowl_trial(params: dict, render: bool = False):
    return run_sort_trial(params, render=render, sim_cls=BowlSim, api=bowl_api, prefix="bowl-")
