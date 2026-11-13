"""
hardware/so101_astra_loop.py

Placeholder skeleton for driving two real SO-101 arms with GPT-6 Astra as the
high-level policy. This is the hardware counterpart of the sim "astra-live"
row in runner/policies.py: same idea (one Astra decision per step, local code
executes it), but on the physical arms via LeRobot instead of Ditto/MuJoCo. Every hardware and API call is stubbed so the file runs
end to end in DRY_RUN mode with no arms, no cameras and no API key.

Architecture (one iteration of the loop):
    1. capture a frame from each camera
    2. read joint positions from each arm
    3. send frames + joint state + task text + short history to Astra
    4. Astra returns a JSON action: target end-effector pose per arm,
       gripper open/closed, and a done flag
    5. validate and clamp the action
    6. IK -> joint targets -> send to servos
    7. wait for motion to settle, log everything, repeat

Fill in the TODOs in this order:
    read_joint_state / send_joint_targets (lerobot, feetech)
    capture_frame (OpenCV)
    solve_ik (URDF + ikpy or lerobot kinematics)
    call_astra (real API client)

Run:  python -m hardware.so101_astra_loop --task "put the red block in the bowl" --dry-run
Logs: results/runs/hw_<timestamp>/step_NNN.json (results/runs is gitignored)
"""

#RUCHA'S CODE REVIEW

#THERE ARE ARM PORTS AND CAMERA IDS FOR THE LKEFT AND RIGHT ARMS, THE CODE IS SETUP TO RUN THE ASTRA MODEL ON THE REAL ARMS, BUT IT IS CURRENTLY IN DRY RUN MODE. THE CODE TAKES IN A TASK AND THEN RUNS THE TASK ON THE ARMS, IT LOGS EVERYTHING TO A LOG DIRECTORY.

#PURPOSE: 
#DEPENDENCIES: From the sim package, we are using the sort_api, mj_sim, mj_sort_sim, and mj_trial modules. We are also using the os, time, json, and signal modules from the standard library. 
# The code also uses the hashlib and importlib.util modules for hashing and dynamic module loading (REF!)
#USED BY: 





from __future__ import annotations

import argparse
import base64
import json
import os
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import numpy as np

# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------

ARM_NAMES = ["left", "right"]

# Serial ports for the two Feetech buses. Pin these with udev rules so they
# do not swap after a reboot.
ARM_PORTS = {
    "left": "/dev/ttyACM0",
    "right": "/dev/ttyACM1",
}

# OpenCV device indices (or /dev/video paths). One camera per arm.
CAMERA_IDS = {
    "left": 0,
    "right": 2,
}

# LeRobot calibration files, produced once per arm by lerobot-calibrate.
CALIB_DIR = Path(__file__).resolve().parent / "calibration"

JOINT_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]

# Joint limits in degrees. TODO: replace with the real limits from calibration.
JOINT_LIMITS_DEG = {
    "shoulder_pan": (-110.0, 110.0),
    "shoulder_lift": (-100.0, 100.0),
    "elbow_flex": (-100.0, 100.0),
    "wrist_flex": (-100.0, 100.0),
    "wrist_roll": (-160.0, 160.0),
    "gripper": (0.0, 100.0),
}

# Workspace box in metres, in each arm's base frame. Anything outside is rejected.
WORKSPACE_MIN = np.array([0.05, -0.30, 0.01])
WORKSPACE_MAX = np.array([0.40, 0.30, 0.30])

MAX_STEP_M = 0.06        # largest allowed end-effector move per decision
MAX_JOINT_DELTA_DEG = 25  # largest allowed joint move per decision
SETTLE_S = 0.8            # wait after sending a command before re-observing
MAX_ITERS = 60
HISTORY_LEN = 4

ASTRA_MODEL = os.environ.get("GAPHUNTER_MODEL", "gpt-6-astra")  # same id the sim uses

# ----------------------------------------------------------------------------
# Data types
# ----------------------------------------------------------------------------


@dataclass
class ArmAction:
    xyz: list[float]           # metres, arm base frame
    rpy: list[float]           # radians (roll, pitch, yaw)
    gripper: str               # "open" or "closed"


@dataclass
class AstraAction:
    left: ArmAction
    right: ArmAction
    done: bool
    reasoning: str = ""


@dataclass
class Observation:
    t: float
    frames: dict[str, np.ndarray]
    joints_deg: dict[str, dict[str, float]]


@dataclass
class HistoryItem:
    action: dict[str, Any]
    outcome: str


# ----------------------------------------------------------------------------
# Hardware stubs
# ----------------------------------------------------------------------------


class Arm:
    """Thin wrapper around a LeRobot SO-101 follower."""

    def __init__(self, name: str, port: str, dry_run: bool):
        self.name = name
        self.port = port
        self.dry_run = dry_run
        self._sim_joints = {j: 0.0 for j in JOINT_NAMES}
        if not dry_run:
            # TODO: wire up lerobot. Roughly:
            # from lerobot.robots.so_follower import SOFollower, SOFollowerConfig
            # cfg = SOFollowerConfig(port=port, id=name, calibration_dir=CALIB_DIR)
            # self.robot = SOFollower(cfg); self.robot.connect()
            raise NotImplementedError("connect the real arm here")

    def read_joint_state(self) -> dict[str, float]:
        if self.dry_run:
            return dict(self._sim_joints)
        # TODO: obs = self.robot.get_observation(); return {j: obs[f"{j}.pos"] ...}
        raise NotImplementedError

    def send_joint_targets(self, targets_deg: dict[str, float]) -> None:
        if self.dry_run:
            self._sim_joints.update(targets_deg)
            return
        # TODO: self.robot.send_action({f"{j}.pos": v for j, v in targets_deg.items()})
        raise NotImplementedError

    def disconnect(self) -> None:
        if not self.dry_run:
            # TODO: self.robot.disconnect()
            pass


class Camera:
    def __init__(self, name: str, device: int | str, dry_run: bool):
        self.name = name
        self.device = device
        self.dry_run = dry_run
        if not dry_run:
            import cv2  # noqa: F401
            # TODO: self.cap = cv2.VideoCapture(device); set 640x480
            raise NotImplementedError("open the real camera here")

    def capture_frame(self) -> np.ndarray:
        if self.dry_run:
            return np.zeros((480, 640, 3), dtype=np.uint8)
        # TODO: ok, frame = self.cap.read(); return frame
        raise NotImplementedError

    def release(self) -> None:
        if not self.dry_run:
            # TODO: self.cap.release()
            pass


# ----------------------------------------------------------------------------
# Kinematics stubs
# ----------------------------------------------------------------------------


def forward_kinematics(joints_deg: dict[str, float]) -> tuple[np.ndarray, np.ndarray]:
    """Return (xyz, rpy) of the gripper. TODO: real FK from the SO-101 URDF."""
    return np.array([0.20, 0.0, 0.10]), np.array([0.0, 0.0, 0.0])


def solve_ik(target_xyz: np.ndarray, target_rpy: np.ndarray,
             seed_deg: dict[str, float]) -> dict[str, float] | None:
    """Return joint targets in degrees, or None if unreachable.
    TODO: ikpy chain from the SO-101 URDF, seeded with the current joints."""
    # Placeholder: hold current pose.
    return dict(seed_deg)


# ----------------------------------------------------------------------------
# Astra call
# ----------------------------------------------------------------------------

SYSTEM_PROMPT = """You are controlling two SO-101 robot arms (left and right).
Each arm has a camera. You receive one image per arm, the current joint
angles in degrees, and a short history of your previous actions.

Respond ONLY with JSON matching this schema:
{
  "reasoning": "<one or two sentences>",
  "left":  {"xyz": [x, y, z], "rpy": [r, p, y], "gripper": "open" | "closed"},
  "right": {"xyz": [x, y, z], "rpy": [r, p, y], "gripper": "open" | "closed"},
  "done": true | false
}
xyz is metres in that arm's base frame, rpy is radians. Move in small steps
(under 6 cm). If an arm should stay still, repeat its current pose.
Set done to true only when the task is visibly complete."""


def encode_frame_png(frame: np.ndarray) -> str:
    try:
        import cv2
        ok, buf = cv2.imencode(".png", frame)
        return base64.b64encode(buf.tobytes()).decode()
    except ImportError:
        return base64.b64encode(frame.tobytes()[:64]).decode()  # dry-run filler


def build_user_content(task: str, obs: Observation,
                       history: list[HistoryItem]) -> list[dict[str, Any]]:
    state_txt = {
        arm: {j: round(v, 1) for j, v in obs.joints_deg[arm].items()}
        for arm in ARM_NAMES
    }
    content: list[dict[str, Any]] = [
        {"type": "text", "text": f"Task: {task}"},
        {"type": "text", "text": "Current joint angles (deg): " + json.dumps(state_txt)},
        {"type": "text", "text": "Recent history: " + json.dumps([asdict(h) for h in history])},
    ]
    for arm in ARM_NAMES:
        content.append({"type": "text", "text": f"Camera: {arm} arm"})
        content.append({
            "type": "image_url",
            "image_url": {"url": "data:image/png;base64," + encode_frame_png(obs.frames[arm])},
        })
    return content


def call_astra(task: str, obs: Observation, history: list[HistoryItem],
               dry_run: bool) -> AstraAction:
    if dry_run:
        # Fake response: nudge both arms forward a little, finish after 3 steps.
        step = len(history)
        fake = {
            "reasoning": f"dry-run step {step}",
            "left": {"xyz": [0.20 + 0.02 * step, 0.0, 0.10], "rpy": [0, 0, 0], "gripper": "open"},
            "right": {"xyz": [0.20 + 0.02 * step, 0.0, 0.10], "rpy": [0, 0, 0], "gripper": "open"},
            "done": step >= 3,
        }
        return parse_action(fake)

    # TODO: replace with the real client. Shape is OpenAI-style chat completions.
    # from openai import OpenAI
    # client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    # resp = client.chat.completions.create(
    #     model=ASTRA_MODEL,
    #     messages=[{"role": "system", "content": SYSTEM_PROMPT},
    #               {"role": "user", "content": build_user_content(task, obs, history)}],
    #     response_format={"type": "json_object"},
    # )
    # return parse_action(json.loads(resp.choices[0].message.content))
    raise NotImplementedError("wire up the Astra API call")


def parse_action(raw: dict[str, Any]) -> AstraAction:
    def arm(d: dict[str, Any]) -> ArmAction:
        if d["gripper"] not in ("open", "closed"):
            raise ValueError(f"bad gripper value: {d['gripper']}")
        return ArmAction(xyz=[float(v) for v in d["xyz"]],
                         rpy=[float(v) for v in d["rpy"]],
                         gripper=d["gripper"])
    return AstraAction(left=arm(raw["left"]), right=arm(raw["right"]),
                       done=bool(raw["done"]), reasoning=str(raw.get("reasoning", "")))


# ----------------------------------------------------------------------------
# Safety
# ----------------------------------------------------------------------------


def clamp_pose(current_xyz: np.ndarray, target_xyz: np.ndarray) -> np.ndarray:
    """Limit step length and keep inside the workspace box."""
    delta = target_xyz - current_xyz
    norm = np.linalg.norm(delta)
    if norm > MAX_STEP_M:
        delta = delta / norm * MAX_STEP_M
    return np.clip(current_xyz + delta, WORKSPACE_MIN, WORKSPACE_MAX)


def clamp_joints(current: dict[str, float], target: dict[str, float]) -> dict[str, float]:
    out = {}
    for j in JOINT_NAMES:
        lo, hi = JOINT_LIMITS_DEG[j]
        v = float(np.clip(target[j], lo, hi))
        v = float(np.clip(v, current[j] - MAX_JOINT_DELTA_DEG, current[j] + MAX_JOINT_DELTA_DEG))
        out[j] = v
    return out


# ----------------------------------------------------------------------------
# Main loop
# ----------------------------------------------------------------------------


def execute(arm: Arm, action: ArmAction, current_joints: dict[str, float]) -> str:
    cur_xyz, _ = forward_kinematics(current_joints)
    target_xyz = clamp_pose(cur_xyz, np.array(action.xyz))
    joints = solve_ik(target_xyz, np.array(action.rpy), current_joints)
    if joints is None:
        return "ik_failed"
    joints["gripper"] = JOINT_LIMITS_DEG["gripper"][1] if action.gripper == "open" \
        else JOINT_LIMITS_DEG["gripper"][0]
    joints = clamp_joints(current_joints, joints)
    arm.send_joint_targets(joints)
    return "ok"


def run(task: str, dry_run: bool, log_dir: Path) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    arms = {n: Arm(n, ARM_PORTS[n], dry_run) for n in ARM_NAMES}
    cams = {n: Camera(n, CAMERA_IDS[n], dry_run) for n in ARM_NAMES}
    history: list[HistoryItem] = []

    try:
        for it in range(MAX_ITERS):
            obs = Observation(
                t=time.time(),
                frames={n: cams[n].capture_frame() for n in ARM_NAMES},
                joints_deg={n: arms[n].read_joint_state() for n in ARM_NAMES},
            )

            action = call_astra(task, obs, history, dry_run)
            print(f"[{it:02d}] {action.reasoning}")

            outcomes = {
                "left": execute(arms["left"], action.left, obs.joints_deg["left"]),
                "right": execute(arms["right"], action.right, obs.joints_deg["right"]),
            }

            history.append(HistoryItem(
                action={"left": asdict(action.left), "right": asdict(action.right)},
                outcome=json.dumps(outcomes),
            ))
            history = history[-HISTORY_LEN:]

            with open(log_dir / f"step_{it:03d}.json", "w") as f:
                json.dump({"t": obs.t, "joints": obs.joints_deg,
                           "action": asdict(action), "outcomes": outcomes}, f, indent=2)
            # TODO: also save obs.frames as PNGs for the demo video.

            if action.done:
                print("Astra reports task done.")
                break
            time.sleep(SETTLE_S if not dry_run else 0.0)
    finally:
        for a in arms.values():
            a.disconnect()
        for c in cams.values():
            c.release()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--task", required=True)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--log-dir", default=str(Path(__file__).resolve().parents[1] / "results" / "runs"
                                          / ("hw_" + time.strftime("%Y%m%d_%H%M%S"))))
    args = p.parse_args()
    run(args.task, args.dry_run, Path(args.log_dir))
