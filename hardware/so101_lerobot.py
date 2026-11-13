"""
hardware/so101_lerobot.py

The two hardware classes of hardware/so101_astra_loop.py on the real arms.
Arm wraps a LeRobot 0.7 SO101Follower on its Feetech bus. Camera wraps an
OpenCV capture. Constructor signatures and return types match the stubs in
so101_astra_loop.py exactly, so run() can use these two names unchanged.

Joint values are LeRobot degrees: the follower is built with
use_degrees=True, so shoulder_pan .. wrist_roll are degrees and gripper is
0 (closed) to 100 (open), the same convention JOINT_LIMITS_DEG assumes.

Needs the `lerobot` conda env and the calibration files that
lerobot-calibrate writes for ids "left" and "right" in CALIB_DIR.
"""

from __future__ import annotations

import numpy as np

from hardware.so101_astra_loop import CALIB_DIR, JOINT_NAMES


class Arm:
    """A LeRobot SO-101 follower driven in degrees."""

    def __init__(self, name: str, port: str, dry_run: bool):
        self.name = name
        self.port = port
        self.dry_run = dry_run
        self._sim_joints = {j: 0.0 for j in JOINT_NAMES}
        if dry_run:
            return
        from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
        cfg = SO101FollowerConfig(port=port, id=name, calibration_dir=CALIB_DIR,
                                  cameras={}, use_degrees=True)
        self.robot = SO101Follower(cfg)
        self.robot.connect()

    def read_joint_state(self) -> dict[str, float]:
        if self.dry_run:
            return dict(self._sim_joints)
        obs = self.robot.get_observation()
        return {j: float(obs[f"{j}.pos"]) for j in JOINT_NAMES}

    def send_joint_targets(self, targets_deg: dict[str, float]) -> None:
        if self.dry_run:
            self._sim_joints.update(targets_deg)
            return
        self.robot.send_action({f"{j}.pos": float(v) for j, v in targets_deg.items()})

    def disconnect(self) -> None:
        if not self.dry_run:
            self.robot.disconnect()


class Camera:
    """An OpenCV capture that returns 480x640 BGR frames."""

    def __init__(self, name: str, device: int | str, dry_run: bool):
        self.name = name
        self.device = device
        self.dry_run = dry_run
        if dry_run:
            return
        import cv2
        self.cap = cv2.VideoCapture(device)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        if not self.cap.isOpened():
            raise RuntimeError(f"camera {name}: cannot open device {device}")

    def capture_frame(self) -> np.ndarray:
        if self.dry_run:
            return np.zeros((480, 640, 3), dtype=np.uint8)
        ok, frame = self.cap.read()
        if not ok:
            raise RuntimeError(f"camera {self.name}: read failed on device {self.device}")
        return frame

    def release(self) -> None:
        if not self.dry_run:
            self.cap.release()
