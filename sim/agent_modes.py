"""Agent control modes that put the model in the loop: live and fly.

Both modes are a policy `run(api)` like any script in results/policies/, so they share the task
text, the API module and docstrings, the 300-call cap and the transcript schema with the script arms.

live: each turn the model sees the task text, the API source and the full history of API calls and
their results, and replies with up to CALLS_PER_TURN calls as one JSON object, for up to MAX_TURNS.
fly: the harness calls the perception functions once, shows the model the results, and the model
writes `run(api)` for this trial. If the script returns before `api.done()` is true, the model sees
the trace so far and may write a continuation, up to MAX_REWRITES times. The sim is never reset.

The wrapper policy passes a log dict from new_log(); every model call appends one entry to
log["calls"] and the end reason lands in log["end"]. The trial runner stores the dict under the
transcript key "agent".
"""
from __future__ import annotations

import inspect
import json
import math
import re
import time

from sim.mj_sim import MAX_API_CALLS
from sim.write_policy import MODEL, SORT_TASK, TASK, ask_astra, ask_opus

MAX_TURNS = 40
CALLS_PER_TURN = 10
MAX_REWRITES = 3
PERCEPTION = {"sort": [("cube_pose", []), ("bin_pose", ["red"]), ("bin_pose", ["green"]),
                       ("bin_pose", ["blue"])],
              "peg": [("hole_pose", [])]}

LIVE_RULES = """You control the arm live, one turn at a time. `api` is the module below; you may call only these
functions. You have at most {cap} API calls per trial, in at most {turns} turns of at most {per_turn} calls
each. Contact force above 150 N aborts the trial as a collision.

```python
{api_source}
```

Each turn, reply with exactly one JSON object and nothing else, in this form:
{{"calls": [{{"api": "move_to", "args": [[0.45, -0.05, 0.10], 0]}}], "stop": false}}
The calls run in order and each return value is appended to the history below. Set "stop" to true
when the task is complete or you want to give up. Turn {turn} of {turns}; {used} of {cap} API calls used.

History of API calls so far (call number. function(args) -> result):
{history}"""

FLY_RULES = """Write a Python module that defines `def run(api):` for this trial only. It is called once and must
return when finished. `api` is the module below; use only these functions (plus `math`, which you may
import). You have at most {cap} API calls per trial. Contact force above 150 N aborts the trial as a
collision.

```python
{api_source}
```

The perception functions have already been called once for this trial. Their results, which count
toward the call budget, are:
{history}

Reply with the complete module in a single ```python code block, followed by at most three sentences
explaining the strategy."""

FLY_CONTINUE = """Your script returned but `api.done()` is still false. The sim was not reset. This is the trace of
every API call so far (call number. function(args) -> result; tcp is the controlled point in metres):
{trace}

Write a new module with `def run(api):` that continues from the current state. Reply with the complete
module in a single ```python code block."""

REPAIR = "Your last reply could not be used: {error}\nReply again in the required form and nothing else."


def task_of(api) -> str:
    """'sort' for sim.sort_api, 'peg' for sim.api."""
    return "sort" if api.__name__.endswith("sort_api") else "peg"


def new_log(mode: str) -> dict:
    return {"mode": mode, "model": MODEL, "calls": [], "end": None, "model_calls": 0,
            "model_seconds": 0.0}


def _scene(task: str) -> str:
    """The task text up to the mode-specific instructions: the cell description and success rule."""
    template = SORT_TASK if task == "sort" else TASK
    return template.split("\n\nWrite a Python module", 1)[0]


def _public(api) -> dict:
    return {name: fn for name, fn in inspect.getmembers(api, inspect.isfunction)
            if not name.startswith("_")}


def _fmt(value) -> str:
    return json.dumps(value, default=str)


def _history_text(history: list[dict]) -> str:
    if not history:
        return "(none yet)"
    lines = []
    for h in history:
        tail = f"error {h['error']}" if "error" in h else f"-> {_fmt(h['result'])}"
        lines.append(f"{h['i']}. {h['api']}({', '.join(_fmt(a) for a in h['args'])}) {tail}")
    return "\n".join(lines)


def _ask(prompt: str, log: dict) -> str:
    ask = ask_opus if MODEL.startswith("claude") else ask_astra
    t0 = time.time()
    try:
        text, meta = ask(prompt)
    except SystemExit as exc:  # ask_opus exits on a bad stop_reason; keep it an ordinary error
        raise RuntimeError(str(exc)) from exc
    dt = time.time() - t0
    log["calls"].append({"prompt_chars": len(prompt), "reply": text, "usage": meta.get("usage"),
                         "model": meta.get("model"), "latency_s": round(dt, 2)})
    log["model_calls"] += 1
    log["model_seconds"] = round(log["model_seconds"] + dt, 2)
    return text


def _call(api, history: list[dict], name: str, args: list) -> dict:
    """Run one API call and record it. Bad arguments are recorded as an error the model will see;
    the sim's own stops (ApiCapReached, CollisionStop) propagate to the trial runner."""
    entry = {"i": len(history) + 1, "api": name, "args": list(args)}
    try:
        entry["result"] = _public(api)[name](*args)
    except (TypeError, ValueError, KeyError) as exc:
        entry["error"] = f"{type(exc).__name__}: {exc}"
    history.append(entry)
    return entry


def _perceive(api, task: str, history: list[dict]) -> None:
    for name, args in PERCEPTION[task]:
        _call(api, history, name, args)


def _parse_json(text: str, api) -> dict:
    """The one JSON object a live turn must contain; raises ValueError with the reason."""
    body = text.strip()
    match = re.search(r"```(?:json)?\n(.*?)```", body, re.S)
    if match:
        body = match.group(1).strip()
    if not body.startswith("{"):
        start, end = body.find("{"), body.rfind("}")
        if start < 0 or end < start:
            raise ValueError("no JSON object found")
        body = body[start:end + 1]
    try:
        reply = json.loads(body)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON: {exc}") from exc
    if not isinstance(reply, dict) or not isinstance(reply.get("calls"), list):
        raise ValueError('expected {"calls": [...], "stop": false}')
    public = _public(api)
    for c in reply["calls"]:
        if (not isinstance(c, dict) or not isinstance(c.get("api"), str)
                or not isinstance(c.get("args", []), list)):
            raise ValueError(f"each call needs 'api' (string) and 'args' (list): {_fmt(c)}")
        if c["api"] not in public:
            raise ValueError(f"unknown function {c['api']!r}; allowed: {sorted(public)}")
    return reply


def _parse_code(text: str) -> str:
    match = re.search(r"```python\n(.*?)```", text, re.S)
    if not match:
        raise ValueError("no ```python code block in the reply")
    return match.group(1)


def _ask_valid(prompt: str, parse, log: dict):
    """One model call plus one repair request. Returns None when both replies fail to parse."""
    for _ in range(2):
        text = _ask(prompt, log)
        try:
            return parse(text)
        except ValueError as exc:
            log["calls"][-1]["parse_error"] = str(exc)
            prompt = prompt + "\n\n" + REPAIR.format(error=exc)
    return None


def run_live(api, task: str, log: dict) -> None:
    scene, source = _scene(task), inspect.getsource(api)
    history: list[dict] = []
    _perceive(api, task, history)
    for turn in range(1, MAX_TURNS + 1):
        prompt = scene + "\n\n" + LIVE_RULES.format(
            cap=MAX_API_CALLS, turns=MAX_TURNS, per_turn=CALLS_PER_TURN, turn=turn,
            used=len(history), api_source=source, history=_history_text(history))
        reply = _ask_valid(prompt, lambda t: _parse_json(t, api), log)
        if reply is None:
            log["end"] = "bad_output"
            return
        calls = reply["calls"][:CALLS_PER_TURN]
        done = False
        for c in calls:
            entry = _call(api, history, c["api"], c.get("args", []))
            if c["api"] == "done" and entry.get("result") is True:
                done = True
        if not done and not any(c["api"] == "done" for c in calls):
            done = _call(api, history, "done", []).get("result") is True  # harness check, counted
        if done:
            log["end"] = "done"
            return
        if reply.get("stop"):
            log["end"] = "stop"
            return
    log["end"] = "turn_cap"


def _run_module(code: str, api) -> None:
    namespace = {"api": api, "math": math}
    exec(code, namespace)  # the model's module, run exactly as a script policy would be
    if "run" not in namespace:
        raise ValueError("the module does not define run(api)")
    namespace["run"](api)


def run_fly(api, task: str, log: dict) -> None:
    scene, source = _scene(task), inspect.getsource(api)
    history: list[dict] = []
    _perceive(api, task, history)
    prompt = scene + "\n\n" + FLY_RULES.format(cap=MAX_API_CALLS, api_source=source,
                                                history=_history_text(history))
    for _ in range(MAX_REWRITES + 1):
        code = _ask_valid(prompt, _parse_code, log)
        if code is None:
            log["end"] = "bad_output"
            return
        try:
            _run_module(code, api)
        except ValueError as exc:  # no run(api): one more chance through the repair path
            log["calls"][-1]["parse_error"] = str(exc)
            code = _ask_valid(prompt + "\n\n" + REPAIR.format(error=exc), _parse_code, log)
            if code is None:
                log["end"] = "bad_output"
                return
            _run_module(code, api)
        if api.done():  # counted, like a script's own done() check
            log["end"] = "done"
            return
        trace = "\n".join(f"{s['i']}. {s['api']}({_fmt(s['args'])}) -> {_fmt(s['result'])} tcp {s['tcp']}"
                          for s in api._SIM.log)
        prompt = scene + "\n\n" + FLY_CONTINUE.format(trace=trace)
    log["end"] = "rewrite_cap"
