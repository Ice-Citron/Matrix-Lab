# LeIsaac SO-101 PickOrange env: API discovery

Written 2026-11-13 after installing Isaac Sim 6.2 + Isaac Lab 3.0 + LeIsaac
in a fresh conda env and stepping `LeIsaac-SO101-PickOrange-v0` headless with
cameras. Everything below was run, not inferred. The brief's environment
section (Isaac Sim 5.1, existing `leisaac` env) was wrong; this is what exists.

## 1. Environment

| Item | Value |
|------|-------|
| Python | `/home/starforge-sf95/miniconda3/envs/leisaac/bin/python` (CPython 3.12.16) |
| isaacsim | 6.2.0.0 (pip wheel from `https://pypi.nvidia.com`, `isaacsim[all,extscache]`) |
| isaaclab | 3.0.0 (PyPI wheel; bundles `isaaclab_tasks`, `isaaclab_physx`, `isaaclab_newton`, ...) |
| torch / warp-lang / newton | 2.11.0 / 1.17.0 / 1.6.1 |
| gymnasium / numpy | 1.4.0 / 2.3.1 |
| leisaac | 0.4.0, editable install of `~/leisaac/source/leisaac`, upstream commit `24d3bcd` (2026-04-15) plus the patches in section 8 |
| LeIsaac checkout | `~/leisaac` (copy of the scratch clone). `~/leisaac/dependencies/IsaacLab` is a shallow clone of tag `v3.0.0`, used only for docs; the pip wheel is what is imported |
| Scene assets | `~/leisaac/assets/robots/so101_follower.usd` and `~/leisaac/assets/scenes/kitchen_with_orange/` from <https://github.com/LightwheelAI/leisaac/releases/tag/v0.1.0> |
| GPU / driver | RTX 5090 32 GB, driver 595.99.02 |

Install commands that were used (in this order):

```bash
conda create -y -n leisaac python=3.12
OMNI_KIT_ACCEPT_EULA=YES ~/miniconda3/envs/leisaac/bin/pip install \
    "isaacsim[all,extscache]==6.2.0" --extra-index-url https://pypi.nvidia.com
~/miniconda3/envs/leisaac/bin/pip install isaaclab==3.0.0 --extra-index-url https://pypi.nvidia.com
~/miniconda3/envs/leisaac/bin/pip install -e ~/leisaac/source/leisaac
```

pip reports four pin conflicts against `isaacsim-kernel`/`isaacsim-core`
(click, psutil, urllib3, llvmlite); they did not matter in any run below.

Two environment variables matter for every run:

- `OMNI_KIT_ACCEPT_EULA=YES`, otherwise Kit refuses to start.
- `LEISAAC_ASSETS_ROOT=/home/starforge-sf95/leisaac/assets`. LeIsaac resolves
  its assets root from the git root of the **current working directory**
  (`leisaac/utils/constant.py:_detect_git_root`). Run from inside the
  Matrix-Lab repo and it will look for `Project-GapHunter/assets`. Set the
  variable explicitly in the adapter.

## 2. Launch recipe that works headless with cameras

```python
from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser(); AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args(); args.headless = True; args.enable_cameras = True
app = AppLauncher(args).app

import gymnasium as gym, torch
from isaaclab_tasks.utils import parse_env_cfg
import leisaac                                   # registers the tasks

cfg = parse_env_cfg("LeIsaac-SO101-PickOrange-v0", num_envs=1, use_fabric=False)
cfg.use_teleop_device("so101leader")             # actions = absolute joint targets, see section 4
env = gym.make("LeIsaac-SO101-PickOrange-v0", cfg=cfg).unwrapped
obs, info = env.reset()
obs, rew, terminated, truncated, info = env.step(torch.zeros((1, 6), device=env.device))
```

`use_fabric=False` is required. With the default `use_fabric=True` the Isaac
RTX cameras render from a wrong transform (section 7) even though
`sensor.data.pos_w`/`quat_w_*` and the USD prim are correct. With Fabric off
the front camera matches the public LeIsaac dataset's front video frame
(plate left, three oranges centre, robot at top). Steps were also faster
without Fabric in this scene (0.019 s vs 0.081 s per step with two 640x480
cameras).

Physics backend: Isaac Lab 3.0 defaults to Newton. The patched LeIsaac cfg sets
`cfg.sim.physics = PhysxCfg(...)` explicitly, so the env runs on Isaac Sim
PhysX without any `physics=` override.

## 3. Verbatim output of the discovery script

Script: `discover_pickorange.py` (kept outside the repo; the logic is the
launch recipe above plus the prints). Run with `--no_fabric`.

```text
APP_LAUNCH_S 3.9
ISAACLAB_VERSION 3.0.0
REGISTERED True
USE_FABRIC False
ACTIONS_CFG SingleArmActionsCfg(arm_action=JointPositionActionCfg(class_type='isaaclab.envs.mdp.actions.joint_actions:JointPositionAction', asset_name='robot', debug_vis=False, clip=None, joint_names=['shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll'], scale=1.0, offset=0.0, preserve_order=False, use_default_offset=True), gripper_action=JointPositionActionCfg(class_type='isaaclab.envs.mdp.actions.joint_actions:JointPositionAction', asset_name='robot', debug_vis=False, clip=None, joint_names=['gripper'], scale=1.0, offset=0.0, preserve_order=False, use_default_offset=True))
SIM_PHYSICS PhysxCfg PhysxCfg(class_type='isaaclab_physx.physics.physx_manager:PhysxManager', ... bounce_threshold_velocity=0.01, ... friction_correlation_distance=0.00625, ...)
SIM_DT 0.016666666666666666 DECIMATION 1 EPISODE_LENGTH_S 25.0
ENV_CLASS ManagerBasedRLEnv
MAKE_S 2.6
RESET_S 0.1
OBS_GROUPS ['policy', 'subtask_terms']
OBS policy['joint_pos'] shape=(1, 6) dtype=torch.float32 device=cuda:0
OBS policy['joint_vel'] shape=(1, 6) dtype=torch.float32 device=cuda:0
OBS policy['joint_pos_rel'] shape=(1, 6) dtype=torch.float32 device=cuda:0
OBS policy['joint_vel_rel'] shape=(1, 6) dtype=torch.float32 device=cuda:0
OBS policy['actions'] shape=(1, 6) dtype=torch.float32 device=cuda:0
OBS policy['wrist'] shape=(1, 480, 640, 3) dtype=torch.uint8 device=cuda:0
OBS policy['front'] shape=(1, 480, 640, 3) dtype=torch.uint8 device=cuda:0
OBS policy['ee_frame_state'] shape=(1, 7) dtype=torch.float32 device=cuda:0
OBS policy['joint_pos_target'] shape=(1, 6) dtype=torch.float32 device=cuda:0
OBS subtask_terms['pick_orange001'] shape=(1,) dtype=torch.bool value=[False]
OBS subtask_terms['put_orange001_to_plate'] shape=(1,) dtype=torch.bool value=[False]
OBS subtask_terms['pick_orange002'] shape=(1,) dtype=torch.bool value=[False]
OBS subtask_terms['put_orange002_to_plate'] shape=(1,) dtype=torch.bool value=[False]
OBS subtask_terms['pick_orange003'] shape=(1,) dtype=torch.bool value=[False]
OBS subtask_terms['put_orange003_to_plate'] shape=(1,) dtype=torch.bool value=[False]
ACTION_SPACE Box(-inf, inf, (1, 6), float32)
SINGLE_ACTION_SPACE Box(-inf, inf, (6,), float32)
ACTION_MANAGER_TERMS ['arm_action', 'gripper_action']
ACTION_DIMS [5, 1]
JOINT_NAMES ['shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll', 'gripper']
BODY_NAMES ['base', 'shoulder', 'upper_arm', 'lower_arm', 'wrist', 'gripper', 'jaw']
SOFT_JOINT_POS_LIMITS_RAD [[-1.9198598861694336, 1.9198598861694336], [-1.7453298568725586, 1.7453298568725586], [-1.7453298568725586, 1.5707998275756836], [-1.65805983543396, 1.65805983543396], [-2.792530059814453, 2.792530059814453], [-0.17453300952911377, 1.7453298568725586]]
JOINT_POS_LIMITS_RAD [[-1.9198598861694336, 1.9198598861694336], [-1.7453298568725586, 1.7453298568725586], [-1.7453298568725586, 1.5707999467849731], [-1.65805983543396, 1.65805983543396], [-2.792530059814453, 2.792530059814453], [-0.17453299462795258, 1.7453298568725586]]
DEFAULT_JOINT_POS_RAD [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
JOINT_POS_AFTER_RESET_RAD [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
ROBOT_ROOT_POS_W [2.200000047683716, -0.6100000143051147, 0.8899999856948853]
ROBOT_ROOT_QUAT_W_wxyz [0.0, 0.0, 1.0, 0.0]      <- label is wrong: isaaclab 3.0 reports (x, y, z, w); this is a 180 deg yaw
ENV_ORIGINS [[-0.0, 0.0, 0.0]]
OBJ Orange001 class=RigidObject root_pos_w=[2.1376898288726807, -0.3198693096637726, 0.9244327545166016] root_quat_w=[0.0, 0.0, 0.0, 1.0] default_root_state=[2.125587224960327, -0.3036655783653259, 0.9244327545166016, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
OBJ Orange002 class=RigidObject root_pos_w=[2.2419209480285645, -0.3130534589290619, 0.9241049885749817] root_quat_w=[0.0, 0.0, 0.0, 1.0] default_root_state=[2.2492213249206543, -0.3321727216243744, 0.9241049885749817, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
OBJ Orange003 class=RigidObject root_pos_w=[2.1556169986724854, -0.4325677454471588, 0.9230279326438904] root_quat_w=[0.0, 0.0, 0.0, 1.0] default_root_state=[2.1537716388702393, -0.4053714871406555, 0.9230279326438904, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
OBJ Plate class=RigidObject root_pos_w=[2.411907196044922, -0.3152843415737152, 0.937487006187439] root_quat_w=[0.0, 0.0, 0.0, 1.0] default_root_state=[2.4153075218200684, -0.3384336233139038, 0.937487006187439, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
EE_TARGET_FRAMES ['gripper', 'jaw']
EE_TARGET_POS_W [[2.179384469985962, -0.3325268030166626, 1.1568522453308105], [2.1805834770202637, -0.23912671208381653, 1.156052827835083]]
TERMINATION_TERMS ['time_out', 'success']
EVENT_TERMS {'reset': ['reset_all', 'domain_randomize_0', 'domain_randomize_1', 'domain_randomize_2', 'domain_randomize_3', 'domain_randomize_4']}
CAMERA front shape=(1, 480, 640, 3) dtype=torch.uint8 mean=105.99 min=1 max=240
CAMERA wrist shape=(1, 480, 640, 3) dtype=torch.uint8 mean=157.09 min=1 max=233
STEP_S 0.014
STEP_RETURN reward [0.0] terminated [False] truncated [False]
SUCCESS_TERM_VALUE [False]
CAMERA_AFTER_STEP front mean=105.99
CAMERA_AFTER_STEP wrist mean=157.09
STEP_S_AVG_10 0.019
ORANGE001_POS_AFTER_WRITE [2.1755871772766113, -0.3036655783653259, 0.9230276942253113]
```

The `root_pos_w` of the oranges differs from `default_root_state` by up to
3 cm because the reset events randomise them (section 6).

## 4. Actions

`cfg.actions` is `MISSING` until `cfg.use_teleop_device(<device>)` is called;
`gym.make` fails without it. The device string only selects the action terms:

| device | arm_action | gripper_action | action dim | meaning |
|--------|-----------|----------------|-----------|---------|
| `so101leader` | `JointPositionActionCfg`, 5 arm joints | `JointPositionActionCfg`, gripper | 6 | absolute joint position targets in **radians**, order below. `use_default_offset=True` adds the default joint pos, which is all zeros, so the action is the absolute target |
| `keyboard` / `gamepad` | `DifferentialInverseKinematicsActionCfg` (relative EE pose) | relative joint pos | 6+2 | delta EE pose; not useful for Astra |
| `so101_state_machine` | IK with absolute EE pose | binary gripper | 8 | used by LeIsaac's own scripted policy |

Use `so101leader`. Action layout (`env.action_manager.action_term_dim == [5, 1]`):

```
index 0 shoulder_pan   1 shoulder_lift   2 elbow_flex   3 wrist_flex   4 wrist_roll   5 gripper
```

Actuators are implicit PD (`stiffness=17.8`, `damping=0.60`, effort limit 10)
with `decimation=1` and `sim.dt=1/60`, so one `env.step` is one 1/60 s physics
step. The arm needs tens of steps to reach a target; with `shoulder_lift=-1.0`
and `elbow_flex=+1.0` commanded for 60 steps the joints read -0.989 / 1.028 rad.

## 5. Joints and limits

`env.scene["robot"].joint_names` (this is the USD order and also the LeRobot
order, no remapping needed):

| idx | joint | soft limit (rad) | soft limit (deg) | USD limit in LeIsaac (deg) | rest pose for success (deg) |
|----:|-------|------------------|------------------|---------------------------|-----------------------------|
| 0 | shoulder_pan  | [-1.9199, 1.9199] | [-110, 110] | (-110, 110) | 0 +/- 30 |
| 1 | shoulder_lift | [-1.7453, 1.7453] | [-100, 100] | (-100, 100) | -100 +/- 30 |
| 2 | elbow_flex    | [-1.7453, 1.5708] | [-100, 90]  | (-100, 90)  | 90 +/- 30 |
| 3 | wrist_flex    | [-1.6581, 1.6581] | [-95, 95]   | (-95, 95)   | 50 +/- 30 |
| 4 | wrist_roll    | [-2.7925, 2.7925] | [-160, 160] | (-160, 160) | 0 +/- 30 |
| 5 | gripper       | [-0.1745, 1.7453] | [-10, 100]  | (-10, 100)  | -10 +/- 30 |

`soft_joint_pos_limits == joint_pos_limits` because
`soft_joint_pos_limit_factor=1.0`. Default and post-reset joint pos is all
zeros (arm pointing straight up). The rest pose the success term wants is
`SO101_FOLLOWER_REST_POSE_RANGE` in `leisaac/assets/robots/lerobot.py`.
`leisaac/utils/robot_utils.py` has `convert_leisaac_action_to_lerobot` and the
inverse for the LeRobot normalised motor range if the real-arm loop needs it.

Data access in Isaac Lab 3.0: every `asset.data.*` field is a `ProxyArray`;
use `.torch` to get a `torch.Tensor` (`robot.data.joint_pos.torch[0]`).
Quaternions are **(x, y, z, w)** everywhere in 3.0 (cfgs, data, `isaaclab.utils.math`).

## 6. Success, subtasks, and the orange pose

Termination terms: `time_out` (25 s, `truncated`) and `success`
(`leisaac.tasks.pick_orange.mdp.terminations.task_done`, `terminated`).
`task_done` is true only when **all three** oranges are within +/-0.10 m in x
and y of the plate's root position and within +/-0.07 m of its height, **and**
the arm is inside the rest-pose ranges above. Read it per step with
`env.termination_manager.get_term("success")` (bool tensor, shape `(num_envs,)`)
or from `terminated`.

For a one-orange trial the adapter should compute its own success from
`obs["subtask_terms"]`: `pick_orangeNNN` = EE jaw frame within 5 cm of the
orange and gripper joint < 0.60 rad; `put_orangeNNN_to_plate` = orange within
+/-0.10 m x/y of the plate, jaw within 5 cm, gripper > 0.60 rad. Or use the
raw positions: `env.scene["Orange001"].data.root_pos_w.torch` and
`env.scene["Plate"].data.root_pos_w.torch`.

Nominal object placement (from the USD, `default_root_state`, world frame,
metres; `env_origins` is zero for one env):

| object | x | y | z |
|--------|---|---|---|
| Orange001 | 2.1256 | -0.3037 | 0.9244 |
| Orange002 | 2.2492 | -0.3322 | 0.9241 |
| Orange003 | 2.1538 | -0.4054 | 0.9230 |
| Plate | 2.4153 | -0.3384 | 0.9375 |
| robot base | 2.2000 | -0.6100 | 0.8900, yaw 180 deg |

The counter top is at z ~ 0.92. The robot faces +y; oranges are ~0.3 m in front
of the base, the plate is ~0.2 m to the robot's left (+x).

Reset events (`mode="reset"`): `reset_all` (`reset_scene_to_default`), then
`domain_randomize_0..3` = `isaaclab.envs.mdp.reset_root_state_uniform` on
Orange001/002/003/Plate with x,y in +/-0.03 m, and `domain_randomize_4` =
`randomize_camera_uniform` on the front camera (+/-2.5 cm, +/-2.5 deg).
To make a trial deterministic, pass a seed and either set the randomisation
terms to `None` on the cfg before `gym.make`
(`cfg.events.domain_randomize_0 = None`, etc.) or overwrite the orange pose
after `env.reset()`:

```python
orange = env.scene["Orange001"]
state = orange.data.default_root_state.torch.clone()      # (N, 13): pos(3) quat xyzw(4) lin vel(3) ang vel(3)
state[:, :3] += env.scene.env_origins
state[:, 0] += dx; state[:, 1] += dy
# yaw about z, xyzw:
state[:, 3:7] = torch.tensor([0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2)], device=env.device)
orange.write_root_pose_to_sim(state[:, :7])               # 3.0 prefers write_root_pose_to_sim_index(root_pose=..., env_ids=...)
orange.write_root_velocity_to_sim(state[:, 7:])
env.sim.step(); env.scene.update(env.sim.get_physics_dt())
```

This was exercised: after writing `default + 0.05 m` in x the orange reported
`root_pos_w = [2.1756, -0.3037, 0.9230]`. Both reset options were verified:
`env.reset(seed=7)` twice gave Orange001 at `[2.1529, -0.3040, 0.9244]` both
times and seed 8 gave `[2.1262, -0.3106, 0.9244]`; with the five
`domain_randomize_*` terms set to `None` the active reset terms are
`['reset_all']` only and Orange001 sits at its default `[2.1256, -0.3037, 0.9244]`.

## 7. Cameras

Both task cameras are `TiledCameraCfg` 640x480 RGB, uint8, `(N, H, W, 3)`,
rendered by Isaac RTX, non-black headless with `--enable_cameras`.

| camera | prim | pose in world after reset (Fabric off) | what it sees |
|--------|------|----------------------------------------|--------------|
| `front` | `Robot/base/front_camera`, offset pos (0, -0.5, 0.6) in base frame, 78 deg FOV | pos (2.21, -0.10, 1.51), view dir (-0.03, -0.31, -0.95) (looking down and back toward the robot) | plate at left, oranges centre, robot from above at top; matches the dataset video |
| `wrist` | `Robot/gripper/wrist_camera`, 75 deg FOV | moves with the gripper | jaws at bottom, wall paneling when the arm is upright |

The front camera is a good Astra viewpoint as is (top-down-ish over the
workspace). If a 45-degree front view is wanted later, add a
`CameraCfg`/`TiledCameraCfg` to `cfg.scene` with `convention="world"`, set
`renderer_cfg.global_settings.enable_translucency = True` on it (all Isaac RTX
cameras in one process must share the same global settings or `gym.make`
raises), and place it with `sensor.set_world_poses_from_view(eyes, targets)`
after reset. Random camera jitter is in `domain_randomize_4`.

Fabric rendering bug (Isaac Lab 3.0.0 + Isaac Sim 6.2.0 wheel): with
`cfg.sim.use_fabric=True` (the default) every Isaac RTX camera renders from a
transform that is not the one in `sensor.data` nor in the USD prim. A free
camera told to look straight down rendered a horizontal view of the floor and
cabinets; the front camera rendered a 5 cm close-up of a plank (depth image
confirmed 0.055 m on the right half and infinity on the left half). The same
scripts with `use_fabric=False` render correctly. Not found upstream yet; keep
`use_fabric=False` in the adapter.

## 8. Patches applied to ~/leisaac (commit 24d3bcd -> isaaclab 3.0 / isaacsim 6.2)

Full diff: `cd ~/leisaac && git diff -- source` (11 files, +88/-43). Every
change traces to one of these API moves; nothing else was touched.

| file | change | reason |
|------|--------|--------|
| `tasks/template/single_arm_env_cfg.py` | `self.sim.physx.*` -> `self.sim.physics = PhysxCfg(bounce_threshold_velocity=0.01, friction_correlation_distance=0.00625)`; `self.sim.render.enable_translucency` -> `camera.renderer_cfg.global_settings.enable_translucency = True` for wrist and front; camera `OffsetCfg.rot` rewritten from wxyz to xyzw | 3.0 removed `SimulationCfg.physx`/`render`; quaternion order changed |
| `assets/robots/lerobot.py` | `init_state.rot (0,0,0,1)` wxyz -> `(0,0,1,0)` xyzw | same 180 deg yaw in the new order |
| `utils/general_assets.py` | `isaacsim.core.utils.prims.get_prim_at_path` -> `isaaclab.sim.utils.get_prim_at_path`; USD sub-asset rotations emitted as xyzw | `isaacsim.core.utils` does not exist in 6.2 |
| `enhance/envs/mdp/observations.py` | added an `image()` shim with the 2.x signature (3.0 only has the `image_rgb` term class); `.torch` on data reads in `ee_frame_state`, `joint_pos_target` | TorchScript math functions reject `ProxyArray` |
| `enhance/envs/mdp/events.py` | `.torch` on camera data reads; accept `env_ids` as a `slice` | 3.0 passes `slice(None)` on full resets |
| `tasks/pick_orange/mdp/observations.py`, `terminations.py` | `.torch` on `root_pos_w`, `target_pos_w`, `joint_pos` reads | ProxyArray |
| `tasks/template/mdp/__init__.py` | re-export `ActionTermCfg` from `isaaclab.managers` | no longer in `isaaclab.envs.mdp` |
| `devices/action_process.py` | `mdp.DifferentialIKControllerCfg` -> `isaaclab.controllers.DifferentialIKControllerCfg` | same |
| `enhance/envs/direct_rl_env.py` | `SimulationManager` import moved inside the function | the `isaacsim.core.simulation_manager` extension is only importable after Kit enabled it |
| `utils/monkey_patch.py` | `monkey_patch()` is now a no-op | 3.0 already has the TerminationManager fix; the old patch would drop 3.0's `_last_episode_dones` bookkeeping |

Not patched (not on the PickOrange path, will fail if used): bi-arm and LeKiwi
template cfgs (`sim.physx`/`sim.render` lines, wxyz camera offsets),
`fold_cloth` direct env (`sim.render.antialiasing_mode`, `isaacsim.core.prims`
cloth imports), and every other task's `mdp` functions that read `.data.*`
without `.torch`.

## 9. Timings and reliability

| measurement | value |
|-------------|-------|
| `AppLauncher` (Kit) start, warm | 3.9 s (first ever start was 85 s: shader cache) |
| `gym.make` PickOrange, 1 env | 2.6 s |
| `env.reset()` | 0.1 s |
| one `env.step` (two 640x480 RTX cameras, Fabric off) | 0.014 s first, 0.019 s mean over 10 |
| same with Fabric on | 0.017 s first, 0.081 s mean over 10 |

Kit occasionally segfaults during startup (`getenv` inside `XOpenDisplay` on
the telemetry thread, exit code 139) before any Python of ours runs; it
happened once in about twelve launches and a plain retry passed. The trial
runner should treat exit 139 before `REGISTERED` as a retry, not an
`api_error`.

Smoke tests that pass: `from isaacsim import SimulationApp` headless (Kit
110.4.0), and `Isaac-Cartpole` (note: 3.0 task ids have no `-v0` suffix) with
`physics=isaacsim_physx`.
