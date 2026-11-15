# Repository structure

```
Matrix-Lab/
├── matrixlab/                 all code
│   ├── core/                  what a user touches
│   │   ├── api/               public API (empty for now)
│   │   └── web_app/           Streamlit app and the static page builders
│   ├── policies/              everything that can be tested
│   │   ├── action_policies/   models that output actions (VLAs)
│   │   └── agent_policies/    LLM agents: analytical (writes a script) and live (in the loop)
│   ├── evaluator/             sweeps and their analysis
│   │   └── analysis/          coverage maths, heatmaps, the flow demo page
│   ├── simulator/             trial dispatch and simulator drivers
│   ├── task_category/         sample tasks, one folder per task and simulator
│   └── hardware/              real robots and cameras
├── results/                   not in Git; published as the Hugging Face dataset
│                              Matrix-Lab/Matrix-Lab-Preliminary-Experiments
├── submodules/                llms-for-robot-control, mujoco_menagerie
├── docs/
└── README.md  LICENSE  Makefile
```

Run every command from the repo root. Modules are addressed as
`matrixlab.<folder>.<file>`.

## policies/

A policy is anything the evaluator can run in a trial.

- `agent_policies/analytical/write_policy.py`: asks a model once to write
  `def run(api)` for a task. The model comes from `GAPHUNTER_MODEL`
  (default `gpt-6-astra`).
  `python -m matrixlab.policies.agent_policies.analytical.write_policy <version> [sort]`
- `agent_policies/analytical/improve_policy.py`: rewrites a policy from
  the failures of a sweep.
  `python -m matrixlab.policies.agent_policies.analytical.improve_policy <src> <dst> --sweep <csv>`
- `agent_policies/live/anthropic_responses_proxy.py`: lets Claude models
  answer through the Responses API that the live harness calls.
- `action_policies/vla/vla.py`: runs a VLA (MolmoAct2, pi0.5) on the
  sphere-in-basket task.

A written policy is a module with `def run(api)`. `api` is the task's API
module (for example `task_category/peg_in_hole/mujoco/api.py`). Policy files
load from `results/policies/<name>.py`, chosen by `GAPHUNTER_POLICY`.

## evaluator/

- `sample.py`: the placement boxes (`local_box`, `deploy_box`,
  `sort_deploy_box`, `isaac_local_box`, `isaac_deploy_box`) and
  `sample_box(box, n, method, seed)` (Sobol, uniform or grid).
- `batch.py`: runs n trials in parallel and writes
  `results/<sim>_<policy>_<tag>.csv`.
  `GAPHUNTER_SIM=mujoco GAPHUNTER_POLICY=v1 python -m matrixlab.evaluator.batch --box deploy --n 600 --tag deploy --policy astra-script`
- `refine.py`: a second pass over the uncertain cells of a sweep.
- `policies.py`: builds `results/policies.json`, the policy table.
- `analysis/coverage.py`: coverage report, success rates and heatmaps.
- `analysis/flow.py` + `flow_page.html`: builds the flow demo page
  `core/web_app/flow.html`.
- `analysis/improve_figure.py`: the v1 to v4 improvement figure.

## simulator/

- `scripts/trial.py`: `run_trial(params, render) -> TrialResult`. It picks
  the backend from `GAPHUNTER_SIM`: `fake` (default), `mujoco`
  (peg-in-hole), `sort` (bin sorting), `ditto` (sphere in basket), `isaac`
  (SO-101).
- `drivers/`: simulator drivers (empty for now).

## task_category/

Each task has its scene, the API a policy may call, and a trial runner.

| Folder | Simulator | API functions |
|---|---|---|
| `peg_in_hole/mujoco/` | MuJoCo, UR5e | `hole_pose`, `move_to`, `step_down`, `done` |
| `bin_sorting/mujoco/` | MuJoCo, UR5e + suction | `cube_pose`, `bin_pose`, `move_to`, `suction`, `done` |
| `sphere_in_basket/ditto/` | ManiSkill (Ditto), Franka | `object_position`, `basket_position`, `move_to`, `set_gripper`, `done` |
| `so101_orange/isaac_sim/` | Isaac Sim 6.2, SO-101 | camera frames and joint targets (`isaac_so101_api.md`) |

The pose functions in the MuJoCo tasks return the simulator's true pose
plus an error model. The Ditto script API returns the true pose. Only the
Isaac task gives the policy camera images.

## hardware/

- `robot/so101/`: the real SO-101 follower through LeRobot
  (`so101_lerobot.py`), the model-in-the-loop controller (`so101_live.py`,
  modes `check`, `jog`, `run`), and the validation kit (`validate.py`).

## core/

- `web_app/streamlit_app.py`: the app. `make demo`
- `web_app/site.py`, `web_app/fill_docs.py`: build the landing page and the
  README from the result files.
- `api/`: reserved for the public API.
