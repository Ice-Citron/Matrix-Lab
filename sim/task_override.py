"""Apply GapHunter (dx, dy, yaw) offsets to Ditto task 1 (DittoOddGeometry-v1).

The framework builds the env inside a separate ManiSkill worker process, and it
selects the Gym id from ditto.catalog, so a new env id alone is not reachable from
scripts/eval.py. Two entry points therefore exist:

* patch(): wraps OddGeometryEnv.initialize_task in place. sim/_inject/sitecustomize.py
  calls it inside the worker as soon as ditto.tasks.geometry_tasks is imported.
* GapHunterOddGeometryEnv, registered as "GapHunter-OddGeometry-v0", for direct Gym use.

Offsets come from the env var GAPHUNTER_PARAMS (JSON: {"dx": m, "dy": m, "yaw": deg}),
or from the module variable OVERRIDE when it is set in-process.
"""
from __future__ import annotations

import json
import math
import os

ENV_VAR = "GAPHUNTER_PARAMS"
ENV_ID = "GapHunter-OddGeometry-v0"
OVERRIDE: dict | None = None  # in-process alternative to the env var

# Object friction override, identical for local and deploy runs. Ditto easy uses 0.10, on which
# Astra's grasps slipped at nominal. GAPHUNTER_FRICTION=native keeps Ditto's own value.
DEFAULT_FRICTION = "0.5"


def object_friction() -> float | None:
    raw = os.environ.get("GAPHUNTER_FRICTION", DEFAULT_FRICTION)
    return None if raw == "native" else float(raw)


def _params() -> dict:
    if OVERRIDE is not None:
        return OVERRIDE
    raw = os.environ.get(ENV_VAR)
    return json.loads(raw) if raw else {}


def current_offsets() -> tuple[float, float, float]:
    params = _params()
    return float(params.get("dx", 0.0)), float(params.get("dy", 0.0)), float(params.get("yaw", 0.0))


def basket_offsets() -> tuple[float, float]:
    """Basket displacement in metres (domain randomisation of the fixture); zero when absent."""
    params = _params()
    return float(params.get("bx", 0.0)), float(params.get("by", 0.0))


def apply_offsets(env, env_idx) -> None:
    """Move the object after the parent's (optional) jitter. Runs before the progress baseline."""
    import numpy as np
    import torch
    from transforms3d.euler import euler2quat
    from transforms3d.quaternions import qmult

    env.task_spec["gaphunter_object_friction"] = object_friction()  # None = Ditto's native value
    bx, by = basket_offsets()
    if bx != 0.0 or by != 0.0:
        # The basket was built kinematic (see patch) so it can be moved; the success test reads
        # env.basket_origin, so it moves with the walls. Nominal origin kept across resets.
        nominal = getattr(env, "_gaphunter_basket_nominal", None)
        if nominal is None:
            nominal = env._gaphunter_basket_nominal = np.asarray(env.basket_origin, dtype=np.float32).copy()
        origin = (nominal + np.asarray((bx, by, 0.0), dtype=np.float32)).astype(np.float32)
        for fixture in (env.basket_floor, env.basket):
            env.batch_pose(fixture, torch.as_tensor(origin), env_idx=env_idx)
        env.basket_origin = origin
        env.task_spec["goal_position"] = (origin + np.asarray((0, 0, env.basket_floor_z))).tolist()
        env.task_spec["gaphunter_basket_offsets"] = {"bx": bx, "by": by}
    dx, dy, yaw = current_offsets()
    if dx == 0.0 and dy == 0.0 and yaw == 0.0:
        return
    actor = env.odd_object
    p = actor.pose.p[env_idx].detach().cpu().numpy().copy()
    q = actor.pose.q[env_idx].detach().cpu().numpy()  # wxyz
    p[:, 0] += dx
    p[:, 1] += dy
    yaw_q = euler2quat(0.0, 0.0, math.radians(yaw))  # world-frame z rotation
    q = np.asarray([qmult(yaw_q, row) for row in q], dtype=np.float32)
    env.batch_pose(actor, torch.as_tensor(p), torch.as_tensor(q), env_idx=env_idx)
    env.task_spec["gaphunter_offsets"] = {"dx": dx, "dy": dy, "yaw_deg": yaw}


def patch() -> None:
    from ditto.tasks import geometry_tasks
    from ditto.tasks.geometry_tasks import OddGeometryEnv

    if getattr(OddGeometryEnv.initialize_task, "_gaphunter", False):
        return
    friction = object_friction()
    move_basket = any(basket_offsets())
    if friction is not None or move_basket:
        # build_task looks up build_actor as a module global; only the task object and, when the
        # basket is displaced, the two basket fixtures (kinematic instead of static) are changed.
        original_build_actor = geometry_tasks.build_actor

        def build_actor(env, name, *args, **kwargs):
            if name == "odd_geometry" and friction is not None:
                kwargs["friction"] = friction
            if name in ("basket", "basket_floor") and move_basket:
                kwargs["static"], kwargs["kinematic"] = False, True
            return original_build_actor(env, name, *args, **kwargs)

        geometry_tasks.build_actor = build_actor
    original = OddGeometryEnv.initialize_task

    def initialize_task(self, env_idx, options):
        original(self, env_idx, options)
        apply_offsets(self, env_idx)

    initialize_task._gaphunter = True
    OddGeometryEnv.initialize_task = initialize_task


def register() -> None:
    """Register GapHunter-OddGeometry-v0. TASK_ID stays the Ditto id: SuiteTaskEnv looks it up in the catalog."""
    from mani_skill.utils.registration import REGISTERED_ENVS, register_env
    from ditto.tasks.geometry_tasks import OddGeometryEnv

    if ENV_ID in REGISTERED_ENVS:
        return

    @register_env(ENV_ID, max_episode_steps=2000)
    class GapHunterOddGeometryEnv(OddGeometryEnv):
        def initialize_task(self, env_idx, options):
            super().initialize_task(env_idx, options)
            apply_offsets(self, env_idx)

    globals()["GapHunterOddGeometryEnv"] = GapHunterOddGeometryEnv
