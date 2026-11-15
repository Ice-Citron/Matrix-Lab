"""
hardware/so101_live.py

One real SO-101 follower arm driven by a vision model (GPT-6 Astra or Claude Fable 5.2).

Each turn: grab a fresh frame from every camera, read the joint angles, send both to the
model, get back ONE small joint move as JSON, clamp it, execute it slowly, repeat.
No leader arm, no inverse kinematics: the model steers in joint space and corrects from
what it sees.

Run from the Matrix-Lab repo root, inside the `lerobot` conda env. Three modes, use them
in this order the first time:

  1. check  nothing moves. Prints joint angles and saves one frame per camera.
       python hardware/so101_live.py check --port /dev/ttyACM0 --front /dev/video1 --top /dev/video0

  2. jog    you move the arm by typing, to learn which way each joint goes.
       python hardware/so101_live.py jog --port /dev/ttyACM0

  3. run    the model moves the arm on its own.
       python hardware/so101_live.py run --port /dev/ttyACM0 --front /dev/video1 --top /dev/video0 \
           --model claude-fable-5-2 --effort low --task "pick up the sphere and put it in the basket"

Use --model gpt-6-astra for Astra. Add --confirm to be asked before every move. Add
--hints "..." to tell the model what you learned in jog (which way each joint goes).
Add --mock to try the loop with no arm, no cameras and no API key.

Keys are read from llms-for-robot-control/credentials.env (same file the sim uses) or from
the environment: OPENAI_API_KEY (+ optional OPENAI_BASE_URL) for Astra, ANTHROPIC_API_KEY
for Fable.

Every run writes results/runs/hw_<time>/ with the camera frames and steps.json.

Safety: keep one hand near the 12 V power switch. Ctrl+C stops the loop; the arm keeps
holding its pose until you press Enter, then it goes limp, so support it first.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3] if Path(__file__).resolve().parent.name == "so101" \
    else Path.cwd()
CREDENTIALS = ROOT / "submodules" / "llms-for-robot-control" / "credentials.env"

ARM_JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
JOINTS = ARM_JOINTS + ["gripper"]

MAX_DELTA_DEG = 15.0      # largest change per joint per move the model may ask for
MAX_MOVES = 3             # moves the model may chain in one reply (no new photo in between)
LIMIT_MARGIN_DEG = 4.0    # stay this far inside each calibrated end stop
MOVE_SECONDS = 1.2        # time taken to execute one move (interpolated, so it is smooth)
STEP_SECONDS = 0.05       # interpolation step
HISTORY_TURNS = 6
IMAGE_MAX_SIDE = 640
JPEG_QUALITY = 80
FAST = False              # Anthropic fast mode (speed: fast); set by --fast
_SPEED_REPORTED = False


# --------------------------------------------------------------------------- arm

class Follower:
    """The real arm, through LeRobot. Positions are degrees (0 = middle of the calibrated
    range) for the five arm joints and 0..100 for the gripper."""

    def __init__(self, port: str, robot_id: str):
        from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
        cfg = SO101FollowerConfig(port=port, id=robot_id, max_relative_target=25.0)
        self.robot = SO101Follower(cfg)
        self.robot.connect()

    def read(self) -> dict[str, float]:
        obs = self.robot.get_observation()
        return {j: float(obs[f"{j}.pos"]) for j in JOINTS}

    def send(self, target: dict[str, float]) -> None:
        self.robot.send_action({f"{j}.pos": float(v) for j, v in target.items()})

    def limits(self) -> dict[str, tuple[float, float]]:
        out = {j: (-100.0, 100.0) for j in ARM_JOINTS}
        out["wrist_roll"] = (-160.0, 160.0)
        out["gripper"] = (0.0, 100.0)
        try:  # half the calibrated tick range, in degrees (4096 ticks per turn)
            for j in ARM_JOINTS:
                c = self.robot.calibration[j]
                half = (c.range_max - c.range_min) / 2 * 360.0 / 4096.0
                if 5.0 < half < 180.0:
                    out[j] = (-half + LIMIT_MARGIN_DEG, half - LIMIT_MARGIN_DEG)
        except Exception as exc:  # noqa: BLE001
            print(f"(could not read calibration limits, using defaults: {exc})")
        return out

    def release(self) -> None:
        self.robot.disconnect()


class MockFollower:
    def __init__(self, *_):
        self.q = {j: 0.0 for j in JOINTS}
        self.q["gripper"] = 50.0

    def read(self):
        return dict(self.q)

    def send(self, target):
        self.q.update(target)

    def limits(self):
        out = {j: (-90.0, 90.0) for j in ARM_JOINTS}
        out["gripper"] = (0.0, 100.0)
        return out

    def release(self):
        pass


def move_smoothly(arm, target: dict[str, float], seconds: float = MOVE_SECONDS) -> None:
    start = arm.read()
    n = max(1, int(seconds / STEP_SECONDS))
    for i in range(1, n + 1):
        a = i / n
        arm.send({j: start[j] + a * (target[j] - start[j]) for j in JOINTS})
        time.sleep(STEP_SECONDS)
    time.sleep(0.3)  # let it settle before the next photo


# ----------------------------------------------------------------------- cameras

class Camera:
    def __init__(self, name: str, path: str):
        import cv2
        self.cv2, self.name = cv2, name
        self.cap = cv2.VideoCapture(int(path) if path.isdigit() else path)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not self.cap.isOpened():
            raise SystemExit(f"Could not open camera '{name}' at {path}")

    def fresh(self) -> np.ndarray:
        """Newest frame. Old frames queue up while the model is thinking, so throw some away."""
        for _ in range(6):
            self.cap.grab()
        ok, frame = self.cap.read()
        if not ok:
            raise RuntimeError(f"Camera '{self.name}' returned no frame")
        return frame  # BGR

    def close(self):
        self.cap.release()


class MockCamera:
    def __init__(self, name: str, *_):
        self.name = name

    def fresh(self):
        return np.full((480, 640, 3), 90, dtype=np.uint8)

    def close(self):
        pass


def to_jpeg(frame: np.ndarray) -> bytes:
    import cv2
    h, w = frame.shape[:2]
    scale = IMAGE_MAX_SIDE / max(h, w)
    if scale < 1:
        frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    if not ok:
        raise RuntimeError("JPEG encode failed")
    return buf.tobytes()


# ------------------------------------------------------------------------ prompt

def system_prompt(limits, gripper_closed_at: int, cameras: list[str], hints: str = "") -> str:
    closed, opened = gripper_closed_at, 100 - gripper_closed_at
    lim = ", ".join(f"{j} {lo:.0f} to {hi:.0f}" for j, (lo, hi) in limits.items() if j != "gripper")
    known = f"\nWhat the operator measured on this arm (trust this):\n{hints}\n" if hints else ""
    return f"""You control one SO-101 robot arm (five joints and a gripper) clamped to a table.
Each turn you receive fresh camera images ({", ".join(cameras)}), the joint angles, and your recent
moves with the joint angles before and after. You reply with 1 to {MAX_MOVES} small moves.

Joints, in degrees, 0 is the middle of each joint's range:
- shoulder_pan: rotates the whole arm about the vertical axis at the base (swings it left or right).
- shoulder_lift: tilts the upper arm forward or back from the base.
- elbow_flex: bends the forearm relative to the upper arm.
- wrist_flex: tilts the gripper up or down.
- wrist_roll: turns the gripper about its own long axis.
Gripper: a number from 0 to 100, where {closed} is fully closed and {opened} is fully open.

Unless stated below, the sign of each joint (which way positive goes) is not given. Work it out
from the images and the history: make a small move, see what changed, then continue.
{known}
Limits (degrees): {lim}.
Each joint may change by at most {MAX_DELTA_DEG:.0f} degrees per move. Larger requests are cut down.

How to work:
- Your moves run in order with no new photo in between. Chain {MAX_MOVES} moves only while the arm is
  travelling through clearly free space. Near the object, the table or a container, send one move
  and look again.
- Open the gripper before going down to an object. Come down from above.
- Close the gripper to grasp, then lift before moving sideways.
- Keep the gripper and arm clear of the table and of containers' walls.
- Set "done" to true only when the images show the task is complete.
- Do not give up early. Give up only if the object is out of reach or missing, and put the reason in "why".

Reply with JSON only, no other text:
{{"see": "<one sentence: where the gripper and the object are now>",
  "plan": "<one sentence: what this move is for>",
  "moves": [{{"delta_deg": {{"shoulder_pan": 0, "shoulder_lift": 0, "elbow_flex": 0, "wrist_flex": 0, "wrist_roll": 0}}, "gripper": null}}],
  "done": false,
  "give_up": false,
  "why": ""}}
In each move, "delta_deg" is the change to apply to each joint (leave out joints that stay put) and
"gripper" is null to leave it, or a number 0 to 100 to set it."""


def turn_text(task: str, turn: int, max_turns: int, joints, history) -> str:
    lines = [f"Task: {task}", f"Turn {turn + 1} of at most {max_turns}.",
             "Joint angles now: " + json.dumps({j: round(v, 1) for j, v in joints.items()})]
    if history:
        lines.append("Your recent moves, oldest first:")
        for h in history[-HISTORY_TURNS:]:
            lines.append(json.dumps(h))
    else:
        lines.append("This is the first turn. No moves yet.")
    return "\n".join(lines)


# ------------------------------------------------------------------- model calls

def _env_file() -> dict:
    env = {}
    if CREDENTIALS.is_file():
        for line in CREDENTIALS.read_text().splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                k, v = line.split("=", 1)
                env[k.strip()] = v.strip().strip('"').strip("'")
    return env


def _key(name: str) -> str:
    v = _env_file().get(name) or os.environ.get(name)
    if not v:
        raise SystemExit(f"{name} not found in {CREDENTIALS} or the environment")
    return v


def _post(url: str, headers: dict, body: dict, timeout: int = 300) -> dict:
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers=headers)
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode(errors="replace")[:500]
            if attempt == 2 or exc.code in (400, 401, 403, 404):
                raise RuntimeError(f"API error {exc.code}: {detail}") from exc
            print(f"API call failed ({exc.code}); retrying", file=sys.stderr)
        except Exception as exc:  # noqa: BLE001
            if attempt == 2:
                raise
            print(f"API call failed ({exc}); retrying", file=sys.stderr)
        time.sleep(4 * (attempt + 1))
    raise RuntimeError("unreachable")


def ask_openai(model, effort, system, text, images) -> str:
    """Same Responses API call the repo's sim/write_policy.py makes, with images added."""
    env = _env_file()
    base = (env.get("OPENAI_BASE_URL") or os.environ.get("OPENAI_BASE_URL")
            or "https://api.openai.com/v1").rstrip("/")
    content = [{"type": "input_text", "text": text}]
    for label, jpg in images:
        content.append({"type": "input_text", "text": f"Image: {label}"})
        content.append({"type": "input_image",
                        "image_url": "data:image/jpeg;base64," + base64.b64encode(jpg).decode()})
    body = {"model": model, "instructions": system, "input": [{"role": "user", "content": content}],
            "reasoning": {"effort": effort}, "store": False}
    data = _post(f"{base}/responses", {"Authorization": f"Bearer {_key('OPENAI_API_KEY')}",
                                       "Content-Type": "application/json"}, body)
    return data.get("output_text") or "".join(
        c.get("text", "") for item in data.get("output", []) if item.get("type") == "message"
        for c in item.get("content", []))


def ask_anthropic(model, effort, system, text, images) -> str:
    """Same Messages API call the repo's sim/write_policy.py makes, with images added."""
    content = [{"type": "text", "text": text}]
    for label, jpg in images:
        content.append({"type": "text", "text": f"Image: {label}"})
        content.append({"type": "image", "source": {"type": "base64", "media_type": "image/jpeg",
                                                    "data": base64.b64encode(jpg).decode()}})
    global FAST, _SPEED_REPORTED
    body = {"model": model, "max_tokens": 8000, "system": system, "thinking": {"type": "adaptive"},
            "output_config": {"effort": effort}, "messages": [{"role": "user", "content": content}]}
    headers = {"x-api-key": _key("ANTHROPIC_API_KEY"), "anthropic-version": "2023-06-01",
               "content-type": "application/json"}
    url = "https://api.anthropic.com/v1/messages"
    if FAST:
        try:
            data = _post(url, headers, {**body, "speed": "fast"})
        except RuntimeError as exc:  # not available for this key, model or rate limit: carry on at standard speed
            print(f"     fast mode was refused ({str(exc)[:160]}); continuing at standard speed")
            FAST = False
            data = _post(url, headers, body)
    else:
        data = _post(url, headers, body)
    if not _SPEED_REPORTED:
        _SPEED_REPORTED = True
        print(f"     (API reports speed: {(data.get('usage') or {}).get('speed', 'not stated')})")
    if data.get("stop_reason") not in ("end_turn", "stop_sequence"):
        raise RuntimeError(f"Model stopped with {data.get('stop_reason')}")
    return "".join(c.get("text", "") for c in data.get("content", []) if c.get("type") == "text")


def ask_mock(model, effort, system, text, images) -> str:
    turn = int(re.search(r"Turn (\d+)", text).group(1))
    return json.dumps({"see": "mock", "plan": f"mock move {turn}",
                       "moves": [{"delta_deg": {"shoulder_pan": 30, "elbow_flex": -5}, "gripper": None},
                                 {"delta_deg": {"shoulder_pan": 5}, "gripper": 100 if turn == 2 else None}],
                       "done": turn >= 3, "give_up": False, "why": ""})


def ask(model, effort, system, text, images, mock=False) -> str:
    if mock:
        return ask_mock(model, effort, system, text, images)
    fn = ask_anthropic if model.startswith("claude") else ask_openai
    return fn(model, effort, system, text, images)


# ------------------------------------------------------------------ action logic

def parse_reply(text: str) -> dict:
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        raise ValueError(f"No JSON in reply: {text[:200]!r}")
    raw = json.loads(m.group(0))
    raw_moves = raw.get("moves")
    if not isinstance(raw_moves, list):  # also accept the single-move form
        raw_moves = [{"delta_deg": raw.get("delta_deg"), "gripper": raw.get("gripper")}]
    moves = []
    for mv in raw_moves[:MAX_MOVES]:
        if not isinstance(mv, dict):
            continue
        delta = {}
        for j, v in (mv.get("delta_deg") or {}).items():
            if j in ARM_JOINTS and isinstance(v, (int, float)) and v != 0:
                delta[j] = float(np.clip(v, -MAX_DELTA_DEG, MAX_DELTA_DEG))
        g = mv.get("gripper")
        g = float(np.clip(g, 0, 100)) if isinstance(g, (int, float)) and not isinstance(g, bool) else None
        if delta or g is not None:
            moves.append({"delta_deg": delta, "gripper": g})
    return {"see": str(raw.get("see", "")), "plan": str(raw.get("plan", "")), "moves": moves,
            "done": bool(raw.get("done")), "give_up": bool(raw.get("give_up")), "why": str(raw.get("why", ""))}


def target_from(joints, action, limits) -> dict[str, float]:
    target = dict(joints)
    for j, d in action["delta_deg"].items():
        lo, hi = limits[j]
        target[j] = float(np.clip(joints[j] + d, lo, hi))
    if action["gripper"] is not None:
        target["gripper"] = action["gripper"]
    return target


# ------------------------------------------------------------------------- modes

def open_cameras(args):
    cls = MockCamera if args.mock else Camera
    cams = []
    for name in ("front", "top"):
        path = getattr(args, name, None)
        if path:
            cams.append(cls(name, path))
    return cams


def new_run_dir() -> Path:
    d = ROOT / "results" / "runs" / ("hw_" + time.strftime("%Y%m%d_%H%M%S"))
    d.mkdir(parents=True, exist_ok=True)
    return d


def finish(arm, cams, mock):
    for c in cams:
        c.close()
    if not mock:
        try:
            input("\nSupport the arm with your hand, then press Enter to switch the motors off. ")
        except (EOFError, KeyboardInterrupt):
            pass
    arm.release()


def mode_check(args):
    arm = (MockFollower if args.mock else Follower)(args.port, args.id)
    cams = open_cameras(args)
    out = new_run_dir()
    try:
        print("Joint angles:", json.dumps({j: round(v, 1) for j, v in arm.read().items()}))
        print("Limits:", json.dumps({j: [round(a), round(b)] for j, (a, b) in arm.limits().items()}))
        for c in cams:
            p = out / f"check_{c.name}.jpg"
            p.write_bytes(to_jpeg(c.fresh()))
            print(f"Saved {c.name} camera frame to {p}")
        print("Nothing was moved. Open the saved pictures and confirm each camera is the one you expect.")
    finally:
        finish(arm, cams, args.mock)


def mode_jog(args):
    arm = (MockFollower if args.mock else Follower)(args.port, args.id)
    limits = arm.limits()
    print("Type a joint and a change in degrees, for example:  shoulder_pan 10   or   elbow_flex -5")
    print("Gripper takes an absolute value 0 to 100:  gripper 80")
    print("Joints:", ", ".join(JOINTS), "| Enter alone prints the angles | q quits")
    try:
        while True:
            q = arm.read()
            print(json.dumps({j: round(v, 1) for j, v in q.items()}))
            try:
                line = input("> ").strip()
            except EOFError:
                break
            if line in ("q", "quit", "exit"):
                break
            if not line:
                continue
            parts = line.split()
            if len(parts) != 2 or parts[0] not in JOINTS:
                print("Not understood.")
                continue
            try:
                val = float(parts[1])
            except ValueError:
                print("Second word must be a number.")
                continue
            action = {"delta_deg": {}, "gripper": None}
            if parts[0] == "gripper":
                action["gripper"] = float(np.clip(val, 0, 100))
            else:
                action["delta_deg"][parts[0]] = float(np.clip(val, -MAX_DELTA_DEG, MAX_DELTA_DEG))
            move_smoothly(arm, target_from(q, action, limits))
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        finish(arm, [], args.mock)


def mode_run(args):
    arm = (MockFollower if args.mock else Follower)(args.port, args.id)
    cams = open_cameras(args)
    if not cams:
        raise SystemExit("Give at least one camera: --front PATH (and optionally --top PATH)")
    limits = arm.limits()
    system = system_prompt(limits, args.gripper_closed_at, [c.name for c in cams], args.hints)
    out = new_run_dir()
    (out / "system_prompt.txt").write_text(system)
    history, steps, outcome = [], [], "max_turns"
    print(f"Model {args.model}, task: {args.task}\nLogging to {out}")
    try:
        for turn in range(args.max_turns):
            frames = {c.name: to_jpeg(c.fresh()) for c in cams}
            for name, jpg in frames.items():
                (out / f"step_{turn:03d}_{name}.jpg").write_bytes(jpg)
            joints = arm.read()
            text = turn_text(args.task, turn, args.max_turns, joints, history)
            t0 = time.time()
            reply = ask(args.model, args.effort, system, text, [(f"{n} camera", j) for n, j in frames.items()],
                        mock=args.mock)
            dt = time.time() - t0
            try:
                action = parse_reply(reply)
            except Exception as exc:  # noqa: BLE001
                print(f"[{turn:02d}] could not read the model's reply ({exc}); asking again next turn")
                history.append({"turn": turn + 1, "error": "your last reply was not valid JSON"})
                steps.append({"turn": turn, "reply": reply, "error": str(exc)})
                continue
            print(f"\n[{turn:02d}] ({dt:.0f}s) sees: {action['see']}\n     plan: {action['plan']}")
            for mv in action["moves"]:
                print(f"     move: {mv['delta_deg']} gripper: {mv['gripper']}")
            step = {"turn": turn, "seconds": round(dt, 1), "joints_before": joints, "reply": reply, "action": action}
            if action["done"] or action["give_up"]:
                outcome = "done" if action["done"] else "give_up"
                steps.append(step)
                print("Model says:", "task complete." if action["done"] else f"giving up: {action['why']}")
                break
            if not action["moves"]:
                history.append({"turn": turn + 1, "plan": action["plan"], "result": "you sent no move"})
                steps.append(step)
                continue
            if args.confirm and not args.mock:
                ans = input("     Enter = do it, s = skip this move, q = stop: ").strip().lower()
                if ans == "q":
                    outcome = "stopped_by_operator"
                    steps.append(step)
                    break
                if ans == "s":
                    history.append({"turn": turn + 1, "plan": action["plan"], "result": "operator skipped this move"})
                    steps.append({**step, "skipped": True})
                    continue
            for mv in action["moves"]:
                move_smoothly(arm, target_from(arm.read(), mv, limits))
            after = arm.read()
            step["joints_after"] = after
            steps.append(step)
            history.append({"turn": turn + 1, "plan": action["plan"], "moves": action["moves"],
                            "before": {j: round(v, 1) for j, v in joints.items()},
                            "after": {j: round(v, 1) for j, v in after.items()}})
    except KeyboardInterrupt:
        outcome = "interrupted"
        print("\nStopped by Ctrl+C. The arm is holding its pose.")
    finally:
        (out / "steps.json").write_text(json.dumps(
            {"task": args.task, "model": args.model, "effort": args.effort, "outcome": outcome, "steps": steps},
            indent=2))
        print(f"\nOutcome: {outcome}. Log and frames in {out}")
        finish(arm, cams, args.mock)


def main():
    global IMAGE_MAX_SIDE, FAST
    p = argparse.ArgumentParser(description="Drive one SO-101 follower with Astra or Fable 5.2.")
    p.add_argument("mode", choices=["check", "jog", "run"])
    p.add_argument("--port", default="/dev/ttyACM0", help="follower serial port")
    p.add_argument("--id", default="matrix_left", help="calibration id used with lerobot-calibrate")
    p.add_argument("--front", help="front camera path, e.g. /dev/video1")
    p.add_argument("--top", help="overhead camera path, e.g. /dev/video0")
    p.add_argument("--model", default=os.environ.get("GAPHUNTER_MODEL", "claude-fable-5-2"))
    p.add_argument("--effort", default="low", choices=["low", "medium", "high"])
    p.add_argument("--task", default="pick up the sphere and put it in the basket")
    p.add_argument("--max-turns", type=int, default=40)
    p.add_argument("--gripper-closed-at", type=int, choices=[0, 100], default=0,
                   help="which gripper value is fully closed on this arm (check with jog)")
    p.add_argument("--confirm", action="store_true", help="ask before each move (default: run on its own)")
    p.add_argument("--hints", default="", help="what you measured in jog, e.g. 'shoulder_pan positive swings "
                   "the arm to the camera's left; shoulder_lift positive leans it forward'")
    p.add_argument("--fast", action="store_true", help="Claude models only: request fast mode (speed: fast). "
                   "Faster output, higher price")
    p.add_argument("--image-side", type=int, default=IMAGE_MAX_SIDE, help="longest image side sent to the model")
    p.add_argument("--mock", action="store_true", help="no arm, no cameras, no API")
    args = p.parse_args()
    IMAGE_MAX_SIDE = args.image_side
    FAST = args.fast
    {"check": mode_check, "jog": mode_jog, "run": mode_run}[args.mode](args)


if __name__ == "__main__":
    main()