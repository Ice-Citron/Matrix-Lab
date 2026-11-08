"""run_trial for the MuJoCo bin-sorting scene: replay a cached policy on one cube (and bin) layout."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import signal
import time

from sim import sort_api
from sim.mj_sim import ApiCapReached, CollisionStop
from sim.mj_sort_sim import SortSim
from sim.mj_trial import MIN_CLIP_BYTES, RESULTS, ROOT, WALL_LIMIT_S, _WallTimeout, _write_clip

POLICY_VERSION = os.environ.get("GAPHUNTER_POLICY", "sort_astra_v1")


def sort_hash(params: dict) -> str:
    """'sort-' + sha1 of the policy and every param. The prefix keeps sort artifacts apart from peg and Ditto
    ones; the policy name keeps two policies run on the same params from overwriting each other."""
    canon = {k: int(v) if k == "seed" else float(v) for k, v in params.items()}
    canon["policy"] = POLICY_VERSION
    return "sort-" + hashlib.sha1(json.dumps(canon, sort_keys=True).encode()).hexdigest()


def _load_policy():
    path = RESULTS / "policies" / f"{POLICY_VERSION}.py"
    spec = importlib.util.spec_from_file_location(f"gaphunter_policy_{POLICY_VERSION}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _exchange():
    path = RESULTS / "policies" / f"{POLICY_VERSION}.exchange.json"
    return json.loads(path.read_text()) if path.exists() else None


def run_sort_trial(params: dict, render: bool = False):
    from sim.trial import TrialResult

    t0 = time.time()
    h = sort_hash(params)
    sim = SortSim(params, record_every=20 if render else 0)
    sort_api._SIM = sim
    mode, error = "", None

    def _alarm(*_):
        raise _WallTimeout()

    previous = signal.signal(signal.SIGALRM, _alarm)
    signal.alarm(WALL_LIMIT_S)
    try:
        _load_policy().run(sort_api)
    except ApiCapReached:
        mode = "timeout"
    except _WallTimeout:
        mode, error = "timeout", f"wall clock > {WALL_LIMIT_S} s"
    except CollisionStop:
        mode = "collision"
    except Exception as exc:  # the policy itself crashed
        mode, error = "api_error", f"{type(exc).__name__}: {exc}"
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)
        sort_api._SIM = None
    outcome = sim.outcome()
    if outcome["success"] and mode != "collision":
        mode = ""
    elif not mode:
        # wrong_bin: in another colour's bin; drop: lifted but not resting in a bin; miss: never lifted
        mode = "wrong_bin" if outcome["bin"] else "drop" if outcome["lifted"] else "miss"
    clip = None
    if render:
        sim._capture()
        try:
            clip = _write_clip(sim.frames, h)
            if clip and (ROOT / clip).stat().st_size < MIN_CLIP_BYTES:
                (ROOT / clip).unlink(missing_ok=True)
                clip = None  # too small to be a usable video
        except Exception:  # noqa: BLE001 - never fail a trial on rendering
            clip = None
        sim.close_renderer()
    outcome.update(failure_mode=mode, error=error, api_calls=sim.calls, pose_error_mm=sim.pose_error_mm)
    transcript = RESULTS / "transcripts" / f"{h}.json"
    transcript.parent.mkdir(parents=True, exist_ok=True)
    transcript.write_text(json.dumps({"params": dict(params), "policy": POLICY_VERSION,
                                      "policy_exchange": _exchange(), "steps": sim.log,
                                      "outcome": outcome}, indent=1, default=float))
    return TrialResult(dict(params), bool(outcome["success"] and mode == ""), mode, sim.calls,
                       time.time() - t0, clip,
                       str(transcript.relative_to(ROOT)), sim.pose_error_mm)
