"""MuJoCo bin sorting: UR5e from mujoco_menagerie with a suction cup on the flange.

World: table top at z=0. One 40 mm cube starts at NOMINAL_CUBE + (dx, dy), rotated by yaw. Three open bins
(red, green, blue) stand in a row at NOMINAL_BINS; phase-2 params bin_<c>_dx / bin_<c>_dy move each bin.
The seed picks the cube colour: COLOURS[seed % 3]. The trial succeeds when the cube rests in the bin of
its own colour with the suction off.

Suction is a MuJoCo adhesion actuator on the cup body. Perception reuses the peg-in-hole camera model
(sim/mj_sim.py): the position error grows as K_RADIAL * r**3, with r = the object's distance from its own
nominal position, plus per-seed Gaussian noise. Colour is read without error.
"""
from __future__ import annotations

import math

import numpy as np

from sim.mj_sim import (COLLISION_N, K_RADIAL, K_YAW, MAX_API_CALLS, NOISE_POS_BASE, NOISE_POS_FRAC, NOISE_YAW,
                        UR5E, ApiCapReached, CollisionStop, _target_rot, mujoco)

COLOURS = ("red", "green", "blue")
RGBA = {"red": "0.85 0.2 0.2 1", "green": "0.2 0.7 0.3 1", "blue": "0.2 0.35 0.85 1"}
NOMINAL_CUBE = np.array([0.45, -0.05])
NOMINAL_BINS = {"red": np.array([0.25, 0.30]), "green": np.array([0.45, 0.30]), "blue": np.array([0.65, 0.30])}
CUBE_HALF = 0.020
BIN_INNER_HALF = 0.060   # 120 mm inside
BIN_WALL = 0.005
BIN_HEIGHT = 0.060
CUP_RADIUS = 0.015
CUP_LEN = 0.050
PAD_HALF = 0.014         # 28 mm square suction pad
SUCTION_N = 20.0         # total adhesion force at ctrl = 1
TABLE = ((0.10, 0.85), (-0.50, 0.50))  # x and y extent of the table top
N_ARM = 6
LIFTED_Z = CUBE_HALF + 0.02  # cube centre this high counts as lifted


def colour_of(params: dict) -> str:
    return COLOURS[int(params["seed"]) % 3]


def bin_offset(params: dict, colour: str) -> np.ndarray:
    return np.array([params.get(f"bin_{colour[0]}_dx", 0.0), params.get(f"bin_{colour[0]}_dy", 0.0)], float)


def _look(pos, target):
    """xyaxes for a camera at pos looking at target, horizon level."""
    fwd = np.asarray(target, float) - np.asarray(pos, float)
    fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, [0, 0, 1])
    right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    return " ".join(f"{v:.4f}" for v in (*right, *up))


def _bin_xml(colour, xy):
    a, w, h = BIN_INNER_HALF, BIN_WALL / 2, BIN_HEIGHT / 2
    walls = [((a + w, 0), (w, a + 2 * w)), ((-a - w, 0), (w, a + 2 * w)), ((0, a + w), (a, w)), ((0, -a - w), (a, w))]
    geoms = "\n".join(f'<geom type="box" pos="{cx} {cy} {h}" size="{sx} {sy} {h}" rgba="{RGBA[colour]}"/>'
                      for (cx, cy), (sx, sy) in walls)
    return f'''<body name="bin_{colour}" pos="{xy[0]} {xy[1]} 0">
      <geom type="box" pos="0 0 {w}" size="{a} {a} {w}" rgba="0.8 0.8 0.8 1"/>
      {geoms}
    </body>'''


def build_model(params: dict) -> mujoco.MjModel:
    arm = (UR5E / "ur5e.xml").read_text()
    arm = arm.replace('meshdir="assets"', f'meshdir="{UR5E / "assets"}"')
    arm = arm[:arm.index("<keyframe>")] + arm[arm.index("</keyframe>") + len("</keyframe>"):]  # nq changes
    cup = f'''<body name="cup" pos="0 0.1 0" quat="-1 1 0 0">
                    <geom type="cylinder" size="{CUP_RADIUS} {CUP_LEN / 2}" pos="0 0 {CUP_LEN / 2}"
                          rgba="0.15 0.15 0.15 1" contype="0" conaffinity="0" mass="0.05"/>
                    <!-- flat pad: box-box contact spreads the adhesion over the face, so the cube hangs level -->
                    <geom name="cup" type="box" size="{PAD_HALF} {PAD_HALF} 0.002" pos="0 0 {CUP_LEN - 0.002}"
                          rgba="0.1 0.1 0.1 1" friction="1" margin="0.002" gap="0.002" mass="0.005"/>
                    <site name="cup_tip" pos="0 0 {CUP_LEN}" size="0.002"/>
                  </body>
                  <site name="attachment_site"'''
    arm = arm.replace('<site name="attachment_site"', cup, 1)
    body = arm[arm.index(">", arm.index("<mujoco")) + 1: arm.rindex("</mujoco>")]
    colour = colour_of(params)
    cx, cy = NOMINAL_CUBE + (params["dx"], params["dy"])
    (tx0, tx1), (ty0, ty1) = TABLE
    bins = "\n".join(_bin_xml(c, NOMINAL_BINS[c] + bin_offset(params, c)) for c in COLOURS)
    side_pos = (1.15, -0.55, 0.75)
    xml = f'''<mujoco model="gaphunter bin sorting">
  {body}
  <visual><global offwidth="640" offheight="480"/></visual>
  <worldbody>
    <light pos="0.45 0 1.5" dir="0 0 -1" directional="true"/>
    <geom name="floor" type="plane" pos="0 0 -0.75" size="2 2 0.05" rgba="0.25 0.25 0.27 1"/>
    <geom name="table" type="box" pos="{(tx0 + tx1) / 2} {(ty0 + ty1) / 2} -0.02"
          size="{(tx1 - tx0) / 2} {(ty1 - ty0) / 2} 0.02" rgba="0.35 0.3 0.25 1"/>
    {bins}
    <body name="cube" pos="{cx} {cy} {CUBE_HALF}" euler="0 0 {math.radians(params["yaw"])}">
      <freejoint/>
      <geom name="cube" type="box" size="{CUBE_HALF} {CUBE_HALF} {CUBE_HALF}" rgba="{RGBA[colour]}"
            mass="0.05" friction="1"/>
    </body>
    <camera name="side" pos="{' '.join(map(str, side_pos))}" xyaxes="{_look(side_pos, (0.42, 0.12, 0.0))}"/>
  </worldbody>
  <actuator><adhesion name="suction" body="cup" ctrlrange="0 1" gain="{SUCTION_N}"/></actuator>
</mujoco>'''
    return mujoco.MjModel.from_xml_string(xml)


class SortSim:
    def __init__(self, params: dict, record_every: int = 0):
        self.params = dict(params)
        self.colour = colour_of(params)
        self.model = build_model(params)
        self.data = mujoco.MjData(self.model)
        self.scratch = mujoco.MjData(self.model)
        self.rng = np.random.default_rng(int(params["seed"]))
        home = np.array([-1.5708, -1.5708, 1.5708, -1.5708, -1.5708, 0.0])
        self.data.qpos[:N_ARM] = home
        self.data.ctrl[:N_ARM] = home
        mujoco.mj_forward(self.model, self.data)
        self.tip = self.model.site("cup_tip").id
        self.cup_geom = self.model.geom("cup").id
        self.cube_geom = self.model.geom("cube").id
        self.cube_body = self.model.body("cube").id
        self.suction_act = self.model.actuator("suction").id
        self.calls, self.log, self.frames = 0, [], []
        self.pose_error_mm = None
        self.max_force = 0.0
        self.max_cube_z = 0.0
        self.record_every = record_every
        self.renderer = None
        self._step_count = 0
        self.target_xyz = self.data.site_xpos[self.tip].copy()
        self.target_yaw = 0.0
        self._settle(400)
        # start 15 cm above the nominal cube
        self._goto(np.array([*NOMINAL_CUBE, 0.15]), 0.0)

    # ---------------------------------------------------------------- ground truth
    def cube_true(self):
        return self.data.xpos[self.cube_body].copy()

    def bin_true(self, colour):
        return NOMINAL_BINS[colour] + bin_offset(self.params, colour)

    def bin_of_cube(self):
        """Colour of the bin the cube rests in, or None."""
        x, y, z = self.cube_true()
        for c in COLOURS:
            bx, by = self.bin_true(c)
            if max(abs(x - bx), abs(y - by)) < BIN_INNER_HALF and z < BIN_HEIGHT:
                return c
        return None

    def outcome(self) -> dict:
        inside = self.bin_of_cube()
        released = self.data.ctrl[self.suction_act] == 0
        success = inside == self.colour and released
        return {"success": bool(success), "colour": self.colour, "bin": inside,
                "lifted": bool(self.max_cube_z > LIFTED_Z), "suction_on": not released,
                "cube_xyz": [round(float(v), 4) for v in self.cube_true()], "max_force_N": round(float(self.max_force), 1)}

    # ---------------------------------------------------------------- physics helpers
    def _contact_force(self) -> float:
        total = 0.0
        f6 = np.zeros(6)
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            if self.cup_geom in (c.geom1, c.geom2) and c.exclude == 0:
                mujoco.mj_contactForce(self.model, self.data, i, f6)
                total += abs(f6[0])
        return total

    def _step(self, n):
        for _ in range(n):
            # gravity + Coriolis compensation on the arm only, as on the real UR controller
            self.data.qfrc_applied[:N_ARM] = self.data.qfrc_bias[:N_ARM]
            mujoco.mj_step(self.model, self.data)
            self._step_count += 1
            if self.record_every > 0 and self._step_count % self.record_every == 0:
                self._capture()
        self.max_cube_z = max(self.max_cube_z, float(self.cube_true()[2]))
        force = self._contact_force()
        self.max_force = max(self.max_force, force)
        return force

    def _settle(self, n=150):
        return self._step(n)

    def _capture(self):
        if self.record_every < 0:  # rendering failed earlier: stop trying
            return
        try:
            if self.renderer is None:
                self.renderer = mujoco.Renderer(self.model, 240, 320)
            self.renderer.update_scene(self.data, camera="side")
            self.frames.append(self.renderer.render().copy())
        except Exception:  # noqa: BLE001 - a missing clip must never fail a trial
            self.record_every = -1

    def close_renderer(self):
        try:
            if self.renderer is not None:
                self.renderer.close()
        except Exception:  # noqa: BLE001
            pass
        self.renderer = None

    def _ik(self, xyz, yaw, iters=200):
        m, d = self.model, self.scratch
        d.qpos[:] = self.data.qpos
        target_r = _target_rot(yaw)
        jacp, jacr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
        lo, hi = m.jnt_range[:N_ARM, 0], m.jnt_range[:N_ARM, 1]
        limited = m.jnt_limited[:N_ARM].astype(bool)
        for _ in range(iters):
            mujoco.mj_kinematics(m, d)
            mujoco.mj_comPos(m, d)
            pos = d.site_xpos[self.tip]
            rot = d.site_xmat[self.tip].reshape(3, 3)
            err_p = xyz - pos
            err_r = 0.5 * (np.cross(rot[:, 0], target_r[:, 0]) + np.cross(rot[:, 1], target_r[:, 1])
                           + np.cross(rot[:, 2], target_r[:, 2]))
            err = np.concatenate([err_p, err_r])
            if np.linalg.norm(err_p) < 1e-5 and np.linalg.norm(err_r) < 1e-4:
                break
            mujoco.mj_jacSite(m, d, jacp, jacr, self.tip)
            j = np.vstack([jacp, jacr])[:, :N_ARM]
            dq = j.T @ np.linalg.solve(j @ j.T + 1e-4 * np.eye(6), err)
            q = d.qpos[:N_ARM] + np.clip(dq, -0.2, 0.2)
            q[limited] = np.clip(q[limited], lo[limited], hi[limited])
            d.qpos[:N_ARM] = q
        return d.qpos[:N_ARM].copy(), float(np.linalg.norm(err_p))

    def _goto(self, xyz, yaw, ramp_steps=250, settle=150):
        q_goal, residual = self._ik(np.asarray(xyz, float), yaw)
        q_start = self.data.ctrl[:N_ARM].copy()
        force = 0.0
        for k in range(1, ramp_steps + 1):
            self.data.ctrl[:N_ARM] = q_start + (q_goal - q_start) * (k / ramp_steps)
            force = max(force, self._step(1))
        force = max(force, self._settle(settle))
        self.target_xyz, self.target_yaw = np.asarray(xyz, float), yaw
        return force, residual

    def _estimate(self, true_xy, nominal_xy):
        d = np.asarray(true_xy, float) - nominal_xy
        r = float(np.linalg.norm(d))
        sigma = NOISE_POS_BASE + NOISE_POS_FRAC * K_RADIAL * r ** 3
        return nominal_xy + d * (1 + K_RADIAL * r * r) + self.rng.normal(0, sigma, 2)

    # ---------------------------------------------------------------- policy-facing API
    def _count(self, api, args, result, force=None):
        self.calls += 1
        tip = self.data.site_xpos[self.tip]
        self.log.append({"i": self.calls, "api": api, "args": args, "result": result,
                         "tcp": [round(float(v), 4) for v in tip],
                         "force_N": None if force is None else round(float(force), 2)})
        if self.calls > MAX_API_CALLS:
            raise ApiCapReached()
        if self.max_force > COLLISION_N:
            raise CollisionStop()

    def cube_pose(self):
        x, y, z = self.cube_true()
        est_xy = self._estimate((x, y), NOMINAL_CUBE)
        mat = self.data.xmat[self.cube_body].reshape(3, 3)
        yaw = math.atan2(mat[1, 0], mat[0, 0])
        y90 = (yaw + math.pi / 4) % (math.pi / 2) - math.pi / 4  # camera sees a square: yaw mod 90
        est_yaw = y90 + K_YAW * y90 ** 3 + self.rng.normal(0, NOISE_YAW)
        result = ((float(est_xy[0]), float(est_xy[1]), float(z + CUBE_HALF)), float(math.degrees(est_yaw)),
                  self.colour)
        if self.pose_error_mm is None:  # first estimate the policy received
            self.pose_error_mm = round(float(np.linalg.norm(est_xy - np.array([x, y]))) * 1000, 3)
        self._count("cube_pose", {}, [list(result[0]), result[1], result[2]])
        return result

    def bin_pose(self, colour):
        est_xy = self._estimate(self.bin_true(colour), NOMINAL_BINS[colour])
        result = ((float(est_xy[0]), float(est_xy[1]), BIN_HEIGHT), 0.0)
        self._count("bin_pose", {"colour": colour}, [list(result[0]), result[1]])
        return result

    def move_to(self, xyz, yaw_deg):
        force, residual = self._goto(np.asarray(xyz, float), math.radians(float(yaw_deg)))
        self._count("move_to", {"xyz": [float(v) for v in xyz], "yaw": float(yaw_deg)},
                    {"ik_residual_mm": round(residual * 1000, 2)}, force)

    def suction(self, on):
        self.data.ctrl[self.suction_act] = 1.0 if on else 0.0
        force = self._settle(150 if on else 400)  # off: let the cube fall and come to rest
        pair = {self.cup_geom, self.cube_geom}
        holds = bool(on) and any({self.data.contact[i].geom1, self.data.contact[i].geom2} == pair
                                 for i in range(self.data.ncon))
        self._count("suction", {"on": bool(on)}, holds, force)
        return holds

    def done(self):
        sorted_ok = self.bin_of_cube() == self.colour  # bin load cell: the right bin feels the cube
        self._count("done", {}, sorted_ok)
        return sorted_ok
