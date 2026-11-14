# hardware

`so101_astra_loop.py` is the Astra control loop for the real SO-101 arms. `sim/isaac_so101.py`
implements the same `Arm` and `Camera` interface on the LeIsaac kitchen scene in Isaac Sim, so the
loop body is the same code in both places. The reference files below were captured from that sim so
the real camera can be mounted to match the view Astra was tested on.

## Camera placement

Numbers read from the task's front camera config (`~/leisaac/source/leisaac/leisaac/tasks/template/
single_arm_env_cfg.py`, `front: TiledCameraCfg`) and verified against the sensor pose reported by the
running sim (camera world position (2.20, -0.11, 1.49), robot base (2.20, -0.61, 0.89), base yawed
180 degrees). All values are in the arm base frame: origin at the base mount, z up, the arm reaches out
along -y at `shoulder_pan` 0.

| Item | Value |
|------|-------|
| Position relative to the base | 0.50 m in front of the base (on the workspace side, along the arm's reach direction), 0 m sideways, 0.60 m above the base mount plane |
| Orientation | Pointed back toward the base and tilted 71 degrees below the horizontal (19 degrees from straight down), no roll. The arm base appears at the top centre of the image and the counter in front of the arm fills the frame |
| Quaternion in the config | `rot=(-0.9862856, 0.0, 0.0, 0.1650476)` xyzw, `convention="ros"`, `pos=(0.0, -0.5, 0.6)` |
| Lens | 78 degree horizontal field of view (focal length 28.7 mm, horizontal aperture 38.11 mm, pinhole) |
| Image | 640 x 480 RGB at 30 FPS |

The oranges lie 20 to 31 cm in front of the base, up to 7 cm either side of the reach line; the plate
centre is 27 cm in front of the base and 22 cm to the robot's right (image left, `shoulder_pan`
positive). The trial's target orange (Orange001) starts 31 cm in front and 7 cm to the robot's left
(image right, farther from the plate). The sim's
`domain_randomize_4` event jitters this camera by up to 2.5 cm and 2.5 degrees per reset; the Astra
trials run with it disabled.

## Reference files

| File | What |
|------|------|
| `reference/sim_view.png` | One 640 x 480 front-camera frame from the sim with the arm at the rest pose |
| `reference/rest_pose.json` | The rest pose in LeRobot motor units (`shoulder_lift` -100, `elbow_flex` 100, `wrist_flex` 52.6, others 0, gripper 0), the values the sim reached, and the same pose in USD degrees. The task's success term requires the arm inside +/-30 USD degrees of this pose |
