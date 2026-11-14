"""Real run_trial: one Astra episode on the LeIsaac SO-101 PickOrange task with GapHunter offsets.

Runs only under the `leisaac` conda env (Isaac Sim 6.2 + Isaac Lab 3.0, see sim/isaac_so101_api.md).
From any other interpreter (the batch runner's .venv) run_isaac_trial re-execs itself as
`<leisaac python> -B -m sim.isaac_so101 '<params json>' [--render]` and parses the last stdout line,
the same pattern as sim/ditto_trial.py.

One arm, joint space: Astra gets the front camera image, the joint state in LeRobot motor units and a
short history, and replies with six absolute joint targets. The sim steps STEPS_PER_DECISION physics
steps per decision. Success is TARGET_ORANGE (the offset orange) placed: its put_orange00N_to_plate
subtask flag, or the orange resting inside the task's plate box at the end of a decision. It is not
the task's own task_done term (all three oranges plus the rest pose): the brief measures one orange in
the bowl, and task_done is recorded in the transcript outcome as well (team-lead decision, 2026-11-13).

Arm and Camera mirror the classes in hardware/so101_astra_loop.py so the loop body is the same code
on the real arm and in the sim.
"""
from __future__ import annotations

import base64
import hashlib
import json
import math
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from hardware.so101_astra_loop import HISTORY_LEN, JOINT_NAMES, MAX_JOINT_DELTA_DEG, clamp_joints
from sim.write_policy import _credentials

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
ARTIFACT_PREFIX = "isaac-so101-"
TASK_ID = "LeIsaac-SO101-PickOrange-v0"
ISAAC_PYTHON = Path(os.environ.get("GAPHUNTER_ISAAC_PYTHON",
                                   Path.home() / "miniconda3/envs/leisaac/bin/python")).expanduser()
ISAAC_ENV = {"OMNI_KIT_ACCEPT_EULA": "YES", "LEISAAC_ASSETS_ROOT": str(Path.home() / "leisaac/assets")}
MODEL = os.environ.get("GAPHUNTER_MODEL", "gpt-6-astra")
DECISION_CAP = int(os.environ.get("GAPHUNTER_DECISION_CAP", "25"))
TRIAL_TIMEOUT_S = float(os.environ.get("GAPHUNTER_TRIAL_TIMEOUT_S", "600"))
STEPS_PER_DECISION = 30          # 1/60 s physics steps the arm gets to follow each target
KIT_STARTUP_MARGIN_S = 240.0     # cold Kit start (shader cache) can take 85 s; the loop's own timeout fires first
CLIP_EVERY, CLIP_FPS, CLIP_MAX_SIDE, CLIP_MAX_SECONDS = 3, 20, 320, 10.0
SEGFAULT_EXIT = 139              # Kit occasionally segfaults in XOpenDisplay before our code runs

# LeRobot joint name -> index in env.scene["robot"].joint_names (the USD order). The SO-101 USD lists
# the joints in LeRobot order already; _World asserts this at creation.
TARGET_ORANGE = "Orange002"  # the offset orange and the one success is measured on: nearest the plate
JOINT_MAP = {"shoulder_pan": 0, "shoulder_lift": 1, "elbow_flex": 2, "wrist_flex": 3, "wrist_roll": 4,
             "gripper": 5}

# Joint values Astra sees and sends: LeRobot motor units (leisaac.utils.robot_utils converters),
# -100..100 per arm joint, gripper 0 (closed) .. 100 (open). The same numbers drive the real arm.
MOTOR_LIMITS = {"shoulder_pan": (-100.0, 100.0), "shoulder_lift": (-100.0, 100.0),
                "elbow_flex": (-100.0, 100.0), "wrist_flex": (-100.0, 100.0),
                "wrist_roll": (-100.0, 100.0), "gripper": (0.0, 100.0)}
# Centre of SO101_FOLLOWER_REST_POSE_RANGE (USD deg 0, -100, 90, 50, 0, -10) in motor units.
REST_POSE = {"shoulder_pan": 0.0, "shoulder_lift": -100.0, "elbow_flex": 100.0, "wrist_flex": 52.6,
             "wrist_roll": 0.0, "gripper": 0.0}

SYSTEM_PROMPT = """You control one SO-101 robot arm (5 joints + gripper) on a kitchen counter.
You receive one image from a fixed camera above the workspace, the current joint values, the joint
limits, and a short history of your previous decisions. Each decision you output absolute joint
targets; the arm then moves toward them for about half a second before the next image is taken, so
compare the reached joints with what you asked for.

Task: pick up the TARGET orange and put it on the plate. The target is the left-hand one of the two
oranges visible at the start: the one nearest the plate, about 17 cm from the plate centre and about
28 cm in front of the base. The task counts as done when that orange rests on the plate; the other
oranges can stay where they are.

Joint values are LeRobot motor units: -100 .. 100 per joint (roughly degrees), gripper 0 = closed ..
100 = open. Each joint may move at most 25 units per decision; larger steps are clipped.
Camera view: the camera stands on the far side of the counter and looks back down at the arm, so the
arm base is at the TOP CENTRE of the image and the counter in front of the arm fills the rest. The
plate is on the left of the image, the three oranges are in the centre.
Joint effects seen in the image (from the upright start pose, all joints 0):
{joint_guide}
Rest pose: {rest_pose}.

Respond ONLY with JSON matching this schema:
{{
  "reasoning": "<one or two sentences>",
  "joints_deg": {{"shoulder_pan": v, "shoulder_lift": v, "elbow_flex": v, "wrist_flex": v,
                 "wrist_roll": v, "gripper": v}},
  "gripper": "open" | "closed",
  "done": true | false
}}
"gripper" overrides the gripper joint value (open = 100, closed = 0). Set done to true only when the
orange visibly lies on the plate and the gripper has released it."""

# Filled from a probe of the sim: which way each joint moves the gripper in the front image.
JOINT_GUIDE = """- shoulder_pan: positive swings the whole arm toward the plate (image left), negative toward image
  right. 0 points straight ahead (down the image).
- shoulder_lift: positive leans the arm forward and down over the counter (toward the bottom of the
  image); negative leans it back toward the base. 0 is upright.
- elbow_flex: positive bends the elbow (gripper lower and closer to the base); negative straightens it
  (gripper higher and further out).
- wrist_flex: positive points the gripper down toward the counter; negative lifts it.
- wrist_roll: rotates the gripper about its own axis.
- gripper: 100 open, 0 closed; close it on the orange, open it over the plate.
At the start (all joints 0) the arm stands upright under the camera and hides the third orange, which
lies on the counter just in front of the base. Verified example: shoulder_pan 0, shoulder_lift 55,
elbow_flex 20, wrist_flex 60 puts the open gripper tip on the counter about 30 cm in front of the base,
between the two visible oranges; shoulder_pan about +12 aims at the target (left) orange, about -10
at the right one. When the gripper presses on the counter the reached joints stay below the
commanded ones."""


def param_hash(params: dict) -> str:
    canon = {"dx": float(params["dx"]), "dy": float(params["dy"]), "yaw": float(params["yaw"]),
             "seed": int(params["seed"])}
    return hashlib.sha1(json.dumps(canon, sort_keys=True).encode()).hexdigest()


def _in_isaac_env() -> bool:
    return Path(sys.prefix).resolve() == ISAAC_PYTHON.resolve().parents[1]


# ---------------------------------------------------------------- the sim (leisaac env only)

class _World:
    """The one Kit app + env per process, created lazily; Arm and Camera share it."""

    instance: "_World | None" = None

    @classmethod
    def get(cls) -> "_World":
        if cls.instance is None:
            cls.instance = cls()
        return cls.instance

    def __init__(self):
        os.environ.update(ISAAC_ENV)
        from isaaclab.app import AppLauncher
        self.app = AppLauncher({"headless": True, "enable_cameras": True}).app
        import gymnasium as gym
        import torch
        import leisaac  # noqa: F401  registers the tasks
        from isaaclab_tasks.utils import parse_env_cfg
        from leisaac.utils.robot_utils import convert_leisaac_action_to_lerobot, convert_lerobot_action_to_leisaac

        self.torch = torch
        self.to_lerobot, self.to_leisaac = convert_leisaac_action_to_lerobot, convert_lerobot_action_to_leisaac
        cfg = parse_env_cfg(TASK_ID, device="cuda", num_envs=1, use_fabric=False)
        cfg.use_teleop_device("so101leader")  # 6 absolute joint targets in radians
        cfg.sim.use_fabric = False            # with Fabric the RTX cameras render from a wrong transform
        cfg.episode_length_s = 2 * TRIAL_TIMEOUT_S  # the env must never auto-reset mid-trial
        # LeIsaac's teleop recorder opens one shared HDF5 file (/tmp/isaaclab/logs/dataset.hdf5) per
        # process; a second Isaac process on the box then dies on the file lock. Not needed here.
        cfg.recorders = None
        for i in range(5):  # orange / plate / camera randomisation: the trial sets the orange pose itself
            setattr(cfg.events, f"domain_randomize_{i}", None)
        self.env = gym.make(TASK_ID, cfg=cfg).unwrapped
        names = self.env.scene["robot"].joint_names
        assert [names[i] for i in JOINT_MAP.values()] == list(JOINT_MAP), names
        self.action = torch.zeros((1, 6), device=self.env.device)
        self.obs = None
        self.success = self.task_done = self.grasped = self.placed = self.target_placed = False
        self.frames: list[np.ndarray] = []
        self.record_every = 0

    def reset(self, seed: int, dx: float, dy: float, yaw_deg: float) -> None:
        self.obs, _ = self.env.reset(seed=seed)
        orange = self.env.scene[TARGET_ORANGE]
        state = orange.data.default_root_state.torch.clone()  # (1, 13): pos, quat xyzw, lin vel, ang vel
        state[:, :3] += self.env.scene.env_origins
        state[:, 0] += dx
        state[:, 1] += dy
        half = math.radians(yaw_deg) / 2
        state[:, 3:7] = self.torch.tensor([0.0, 0.0, math.sin(half), math.cos(half)], device=self.env.device)
        orange.write_root_pose_to_sim(state[:, :7])
        orange.write_root_velocity_to_sim(state[:, 7:])
        self.action.zero_()
        self.success = self.task_done = self.grasped = self.placed = self.target_placed = False
        self.frames = []
        self.step(1)  # settle one step so the observation shows the new orange pose

    def step(self, n: int) -> None:
        for i in range(n):
            self.obs, _, terminated, _, _ = self.env.step(self.action)
            flags = self.flags()
            self.grasped |= any(v for k, v in flags.items() if k.startswith("pick_"))
            self.placed |= any(v for k, v in flags.items() if k.startswith("put_"))
            self.target_placed |= flags[f"put_{TARGET_ORANGE.lower()}_to_plate"]
            if self.record_every and i % self.record_every == 0:
                self.frames.append(self.front_rgb())
            self.task_done |= bool(terminated[0])  # the only terminated term is task_done (strict, three oranges)
            if self.task_done:
                self.success = True
                return
        # Checked once per decision, after the arm had its steps: a held orange swinging past the plate
        # mid-move must not count.
        if self.target_placed or self.orange_on_plate():
            self.success = True

    def orange_on_plate(self) -> bool:
        """TARGET_ORANGE inside the box task_done uses (+/-0.10 m x/y, +/-0.07 m z of the plate), resting
        (speed below 0.05 m/s) and not held in the gripper (pick flag off)."""
        orange = self.env.scene[TARGET_ORANGE].data
        plate = self.env.scene["Plate"].data
        d = (orange.root_pos_w.torch[0] - plate.root_pos_w.torch[0]).abs().tolist()
        speed = float(orange.root_lin_vel_w.torch[0].norm())
        held = self.flags()[f"pick_{TARGET_ORANGE.lower()}"]
        return d[0] < 0.10 and d[1] < 0.10 and d[2] < 0.07 and speed < 0.05 and not held

    def flags(self) -> dict[str, bool]:
        return {k: bool(v[0]) for k, v in self.obs["subtask_terms"].items()}

    def front_rgb(self) -> np.ndarray:
        return self.obs["policy"]["front"][0].cpu().numpy()

    def joints_lerobot(self) -> dict[str, float]:
        rad = self.obs["policy"]["joint_pos"][0].cpu().numpy()
        units = self.to_lerobot(np.array([[rad[JOINT_MAP[j]] for j in JOINT_NAMES]]))[0]
        return {j: float(units[k]) for k, j in enumerate(JOINT_NAMES)}

    def send(self, targets_deg: dict[str, float]) -> None:
        rad = self.to_leisaac(np.array([[targets_deg[j] for j in JOINT_NAMES]]))[0]
        for k, j in enumerate(JOINT_NAMES):
            self.action[0, JOINT_MAP[j]] = float(rad[k])
        self.step(STEPS_PER_DECISION)

    def positions(self) -> dict[str, list[float]]:
        return {name: [round(float(v), 4) for v in self.env.scene[name].data.root_pos_w.torch[0].cpu()]
                for name in ("Orange001", "Orange002", "Orange003", "Plate")}


class Arm:
    """Same interface as hardware.so101_astra_loop.Arm; port is accepted and ignored."""

    def __init__(self, name: str, port: str, dry_run: bool):
        self.name = name
        self.port = port
        self.dry_run = dry_run
        self._sim_joints = {j: 0.0 for j in JOINT_NAMES}
        self.world = None if dry_run else _World.get()

    def read_joint_state(self) -> dict[str, float]:
        if self.dry_run:
            return dict(self._sim_joints)
        return self.world.joints_lerobot()

    def send_joint_targets(self, targets_deg: dict[str, float]) -> None:
        if self.dry_run:
            self._sim_joints.update(targets_deg)
            return
        self.world.send(targets_deg)

    def disconnect(self) -> None:
        pass


class Camera:
    """Same interface as hardware.so101_astra_loop.Camera; device is accepted and ignored."""

    def __init__(self, name: str, device: int | str, dry_run: bool):
        self.name = name
        self.device = device
        self.dry_run = dry_run
        self.world = None if dry_run else _World.get()

    def capture_frame(self) -> np.ndarray:
        """480x640x3 uint8 BGR (OpenCV convention) from the task's front camera."""
        if self.dry_run:
            return np.zeros((480, 640, 3), dtype=np.uint8)
        return np.ascontiguousarray(self.world.front_rgb()[:, :, ::-1])

    def release(self) -> None:
        pass


# ---------------------------------------------------------------- Astra

def encode_frame_png(frame_bgr: np.ndarray) -> str:
    import cv2
    ok, buf = cv2.imencode(".png", frame_bgr)
    return base64.b64encode(buf.tobytes()).decode()


def system_prompt() -> str:
    return SYSTEM_PROMPT.format(joint_guide=JOINT_GUIDE, rest_pose=json.dumps(REST_POSE))


def build_prompt_text(step: int, joints: dict[str, float], history: list[dict]) -> str:
    return "\n".join([
        f"Decision {step + 1} of {DECISION_CAP}.",
        "Current joints: " + json.dumps({j: round(v, 1) for j, v in joints.items()}),
        "Joint limits: " + json.dumps(MOTOR_LIMITS),
        "Recent history (oldest first): " + json.dumps(history),
    ])


def call_astra(text: str, frame_bgr: np.ndarray) -> tuple[str, dict]:
    """One Responses API call with the image; same credentials, model and retry policy as ask_astra."""
    import urllib.request
    key, base = _credentials()
    body = {"model": MODEL, "instructions": system_prompt(), "reasoning": {"effort": "medium"}, "store": False,
            "input": [{"role": "user", "content": [
                {"type": "input_text", "text": text},
                {"type": "input_image", "detail": "high",
                 "image_url": "data:image/png;base64," + encode_frame_png(frame_bgr)}]}]}
    req = urllib.request.Request(f"{base}/responses", data=json.dumps(body).encode(),
                                 headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=300) as resp:
                data = json.loads(resp.read())
            break
        except Exception as exc:  # noqa: BLE001
            if attempt == 2:
                raise
            print(f"Astra call failed ({exc}); retrying", file=sys.stderr, flush=True)
            time.sleep(5 * (attempt + 1))
    reply = data.get("output_text") or "".join(
        c.get("text", "") for item in data.get("output", []) if item.get("type") == "message"
        for c in item.get("content", []))
    return reply, {"model": data.get("model", MODEL), "usage": data.get("usage")}


def parse_joint_action(text: str) -> dict:
    """Astra's reply -> {"reasoning", "joints_deg" (six joints), "gripper", "done"}. Raises on bad JSON."""
    raw = json.loads(text[text.index("{"): text.rindex("}") + 1])
    gripper = raw["gripper"]
    if gripper not in ("open", "closed"):
        raise ValueError(f"bad gripper value: {gripper}")
    joints = {j: float(raw["joints_deg"][j]) for j in JOINT_NAMES if j != "gripper"}
    joints["gripper"] = MOTOR_LIMITS["gripper"][1] if gripper == "open" else MOTOR_LIMITS["gripper"][0]
    return {"reasoning": str(raw.get("reasoning", "")), "joints_deg": joints, "gripper": gripper,
            "done": bool(raw["done"])}


# ---------------------------------------------------------------- trial (leisaac env)

def _write_transcript(payload: dict, h: str) -> str:
    path = RESULTS / "transcripts" / f"{ARTIFACT_PREFIX}{h}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=1, default=str))
    return str(path.relative_to(ROOT))


def _write_clip(frames: list[np.ndarray], h: str) -> str | None:
    """Front-camera frames -> mp4, <=320 px, <=10 s (uniform subsample), like ditto_trial."""
    if not frames:
        return None
    import imageio.v2 as imageio
    max_frames = int(CLIP_MAX_SECONDS * CLIP_FPS)
    if len(frames) > max_frames:
        idx = np.linspace(0, len(frames) - 1, max_frames).astype(int)
        frames = [frames[i] for i in idx]
    hgt, wid = frames[0].shape[:2]
    step = max(1, int(np.ceil(max(hgt, wid) / CLIP_MAX_SIDE)))
    frames = [f[::step, ::step] for f in frames]
    frames = [f[: f.shape[0] // 2 * 2, : f.shape[1] // 2 * 2] for f in frames]  # even dims for h264
    path = RESULTS / "clips" / f"{ARTIFACT_PREFIX}{h}.mp4"
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimwrite(str(path), frames, fps=CLIP_FPS, macro_block_size=1)
    return str(path.relative_to(ROOT))


def _safe_clip(frames: list[np.ndarray], h: str) -> str | None:
    try:
        return _write_clip(frames, h)
    except Exception as exc:  # a missing clip must not fail the trial
        print(f"[gaphunter] clip export failed: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        return None


def _run_in_process(params: dict, render: bool) -> dict:
    t0 = time.time()
    h = param_hash(params)
    world = _World.get()
    world.reset(int(params["seed"]), float(params["dx"]), float(params["dy"]), float(params["yaw"]))
    world.record_every = CLIP_EVERY if render else 0
    arm, cam = Arm("so101", "", dry_run=False), Camera("front", 0, dry_run=False)
    decisions: list[dict] = []
    history: list[dict] = []
    mode = ""
    for it in range(DECISION_CAP):
        if time.time() - t0 > TRIAL_TIMEOUT_S:
            mode = "timeout"
            break
        frame, joints = cam.capture_frame(), arm.read_joint_state()
        text = build_prompt_text(it, joints, history)
        row = {"step": it, "prompt": text, "joints_before": joints, "t_start": round(time.time() - t0, 1)}
        try:
            reply, meta = call_astra(text, frame)
        except Exception as exc:  # noqa: BLE001  retries exhausted: the trial ends as api_error
            row["error"] = f"{type(exc).__name__}: {exc}"
            decisions.append(row)
            mode = "api_error"
            break
        row.update(reply=reply, usage=meta.get("usage"), astra_s=round(time.time() - t0 - row["t_start"], 1))
        try:
            action = parse_joint_action(reply)
        except (ValueError, KeyError, TypeError) as exc:
            row["error"] = f"unparseable reply: {type(exc).__name__}: {exc}"
            decisions.append(row)
            history = (history + [{"error": "unparseable reply, arm did not move"}])[-HISTORY_LEN:]
            continue
        target = clamp_joints(joints, action["joints_deg"])
        # Sim only: the gripper is not rate-limited, so open->closed is one decision, not four.
        # hardware/so101_astra_loop.clamp_joints is unchanged; revisit for the real arm.
        target["gripper"] = float(np.clip(action["joints_deg"]["gripper"], 0.0, 100.0))
        arm.send_joint_targets(target)
        after = arm.read_joint_state()
        row.update(action=action, target_sent=target, joints_after=after, flags=world.flags(),
                   positions=world.positions(), t_end=round(time.time() - t0, 1))
        decisions.append(row)
        history = (history + [{"reasoning": action["reasoning"],
                               "target": {j: round(v, 1) for j, v in target.items()},
                               "reached": {j: round(v, 1) for j, v in after.items()}}])[-HISTORY_LEN:]
        print(f"[{it:02d}] {action['reasoning']}", flush=True)
        if world.success or action["done"]:
            break
    success = world.success
    if success:
        mode = ""
    elif not mode:  # cap hit or Astra said done early: miss = never grasped, drop = grasped but not placed
        mode = "miss" if not world.grasped else "drop"
    t_wall = time.time() - t0
    outcome = {"success": success, "failure_mode": mode, "task_done_three_oranges": world.task_done,
               "target_orange": TARGET_ORANGE, "target_on_plate": world.orange_on_plate(),
               "target_put_flag": world.target_placed, "grasped": world.grasped, "placed": world.placed,
               "flags_final": world.flags(), "positions_final": world.positions(),
               "n_decisions": len(decisions), "t_wall": round(t_wall, 1)}
    transcript = _write_transcript({"params": dict(params), "model": MODEL, "decision_cap": DECISION_CAP,
                                    "steps_per_decision": STEPS_PER_DECISION, "max_joint_delta": MAX_JOINT_DELTA_DEG,
                                    "system_prompt": system_prompt(), "decisions": decisions,
                                    "outcome": outcome}, h)
    return {
        "params": dict(params),
        "success": success,
        "failure_mode": mode,
        "n_decisions": len(decisions),
        "t_wall": t_wall,
        "clip": _safe_clip(world.frames, h) if render else None,
        "transcript": transcript,
        "pose_error_mm": None,
    }


# ---------------------------------------------------------------- entry point used by sim.trial

def run_isaac_trial(params: dict, render: bool = False) -> dict:
    if _in_isaac_env():
        return _run_in_process(params, render)
    return _run_in_subprocess(params, render)


def _run_in_subprocess(params: dict, render: bool) -> dict:
    t0 = time.time()
    clean = {"dx": float(params["dx"]), "dy": float(params["dy"]), "yaw": float(params["yaw"]),
             "seed": int(params["seed"])}
    cmd = [str(ISAAC_PYTHON), "-B", "-m", "sim.isaac_so101", json.dumps(clean)]
    if render:
        cmd.append("--render")
    env = {**os.environ, **ISAAC_ENV}
    for attempt in range(2):  # retry once on the Kit startup segfault
        proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                text=True, start_new_session=True)
        try:
            stdout, stderr = proc.communicate(timeout=TRIAL_TIMEOUT_S + KIT_STARTUP_MARGIN_S)
            mode = "api_error"  # crash or no result line
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            stdout, stderr = proc.communicate()
            mode = "timeout"
        lines = [line for line in stdout.splitlines() if line.startswith("GAPHUNTER_RESULT ")]
        if lines:  # Kit teardown may crash after the result was printed; the result still stands
            return json.loads(lines[-1].removeprefix("GAPHUNTER_RESULT "))
        if proc.returncode != SEGFAULT_EXIT or attempt == 1:
            break
        print(f"[gaphunter] Kit segfaulted at startup (exit {SEGFAULT_EXIT}); retrying once", file=sys.stderr, flush=True)
    log = RESULTS / "transcripts" / f"{ARTIFACT_PREFIX}{param_hash(params)}.stderr.txt"
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(f"mode={mode} returncode={proc.returncode}\n" + stdout[-20000:]
                   + "\n---- stderr ----\n" + stderr[-20000:])
    return {"params": dict(params), "success": False, "failure_mode": mode, "n_decisions": 0,
            "t_wall": time.time() - t0, "clip": None, "transcript": None, "pose_error_mm": None}


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))
    params = json.loads(sys.argv[1])
    result = _run_in_process(params, render="--render" in sys.argv[2:])
    print("GAPHUNTER_RESULT " + json.dumps(result), flush=True)
    try:
        _World.instance.app.close()
    except Exception as exc:  # noqa: BLE001  Kit teardown must not turn a finished trial into an error
        print(f"[gaphunter] app.close failed: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
    os._exit(0)
