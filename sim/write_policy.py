"""Ask a model once to write policy.py against the task's API; cache it as results/policies/<version>.py.

  .venv/bin/python -m sim.write_policy v1                    # peg-in-hole, Astra (sim/api.py)
  .venv/bin/python -m sim.write_policy sort_astra_v1 sort    # bin sorting (sim/sort_api.py)
  GAPHUNTER_MODEL=claude-opus-5-5 .venv/bin/python -m sim.write_policy sort_opus_v1 sort
"""
from __future__ import annotations

import inspect
import json
import os
import re
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CREDENTIALS = ROOT / "llms-for-robot-control" / "credentials.env"
MODEL = os.environ.get("GAPHUNTER_MODEL", "gpt-6-astra")

TASK = """You are writing the control policy for a peg-in-hole insertion cell.

A UR5e arm holds a square peg (20 x 20 mm, 80 mm long) pointing straight down. A block on the table
has a square hole (21 x 21 mm, 0.5 mm clearance per side, 50 mm deep); its top surface is at z = 0.05 m.
In our test cell the hole is placed at x = 0.45 +/- 0.05 m, y = 0.00 +/- 0.05 m, yaw = 0 +/- 15 degrees.
The trial succeeds when the peg is seated at least 30 mm deep in the hole with less than 1 mm lateral error.

Write a Python module that defines `def run(api):`. It is called once per trial and must return when
finished. `api` is the module below; use only these four functions (plus `math`, which you may import).
You have at most 300 API calls per trial. Contact force above 150 N aborts the trial as a collision.

```python
{api_source}
```

Reply with the complete module in a single ```python code block, followed by at most three sentences
explaining the strategy."""

SORT_TASK = """You are writing the control policy for a bin-sorting cell.

A UR5e arm has a suction cup pointing straight down. One 40 mm cube lies on the table; it is red, green or blue.
Three open bins stand on the table, one per colour; each is 120 x 120 mm inside, with walls 60 mm high.
In our test cell the cube is placed at x = 0.45 +/- 0.05 m, y = -0.05 +/- 0.05 m, yaw = 0 +/- 15 degrees.
The bins are fixed at red (0.25, 0.30), green (0.45, 0.30) and blue (0.65, 0.30).
The trial succeeds when the cube rests inside the bin of its own colour with the suction off.

Write a Python module that defines `def run(api):`. It is called once per trial and must return when
finished. `api` is the module below; use only these five functions (plus `math`, which you may import).
You have at most 300 API calls per trial. Contact force on the pad above 150 N aborts the trial as a collision.

```python
{api_source}
```

Reply with the complete module in a single ```python code block, followed by at most three sentences
explaining the strategy."""


def _env():
    env = {}
    for line in CREDENTIALS.read_text().splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            k, v = line.split("=", 1)
            env[k.strip()] = v.strip().strip('"').strip("'")
    return env


def _credentials():
    env = _env()
    return env.get("OPENAI_API_KEY") or os.environ["OPENAI_API_KEY"], \
        env.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")


def ask_astra(prompt: str) -> tuple[str, dict]:
    key, base = _credentials()
    body = {"model": MODEL, "input": prompt, "reasoning": {"effort": "medium"}, "store": False}
    req = urllib.request.Request(f"{base}/responses", data=json.dumps(body).encode(),
                                 headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    for attempt in range(3):  # retry 2x with backoff
        try:
            with urllib.request.urlopen(req, timeout=300) as resp:
                data = json.loads(resp.read())
            break
        except Exception as exc:  # noqa: BLE001
            if attempt == 2:
                raise
            print(f"Astra call failed ({exc}); retrying", file=sys.stderr)
            time.sleep(5 * (attempt + 1))
    text = data.get("output_text") or "".join(
        c.get("text", "") for item in data.get("output", []) if item.get("type") == "message"
        for c in item.get("content", []))
    return text, {"model": data.get("model", MODEL), "usage": data.get("usage")}


def ask_opus(prompt: str) -> tuple[str, dict]:
    """Same single-shot call on the Anthropic Messages API (raw HTTP: the SDK is not a project dependency).
    Effort medium matches Astra's reasoning effort; no server-side fallback, so the named model wrote the policy."""
    key = _env().get("ANTHROPIC_API_KEY") or os.environ["ANTHROPIC_API_KEY"]
    body = {"model": MODEL, "max_tokens": 16000, "thinking": {"type": "adaptive"},
            "output_config": {"effort": "medium"}, "messages": [{"role": "user", "content": prompt}]}
    req = urllib.request.Request("https://api.anthropic.com/v1/messages", data=json.dumps(body).encode(),
                                 headers={"x-api-key": key, "anthropic-version": "2023-06-01",
                                          "content-type": "application/json"})
    for attempt in range(3):  # retry 2x with backoff
        try:
            with urllib.request.urlopen(req, timeout=600) as resp:
                data = json.loads(resp.read())
            break
        except Exception as exc:  # noqa: BLE001
            if attempt == 2:
                raise
            print(f"Opus call failed ({exc}); retrying", file=sys.stderr)
            time.sleep(5 * (attempt + 1))
    if data.get("stop_reason") != "end_turn":
        raise SystemExit(f"Opus stopped with {data.get('stop_reason')}: {data.get('stop_details')}")
    text = "".join(c.get("text", "") for c in data.get("content", []) if c.get("type") == "text")
    return text, {"model": data.get("model", MODEL), "usage": data.get("usage")}


def main(version: str = "v1", task: str = "peg"):
    from sim import api, sort_api
    module, template = (sort_api, SORT_TASK) if task == "sort" else (api, TASK)
    prompt = template.replace("{api_source}", inspect.getsource(module))
    ask = ask_opus if MODEL.startswith("claude") else ask_astra
    text, meta = ask(prompt)
    match = re.search(r"```python\n(.*?)```", text, re.S)
    if not match:
        raise SystemExit(f"No code block in the model's reply:\n{text}")
    out = ROOT / "results" / "policies"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{version}.py").write_text(match.group(1))
    (out / f"{version}.exchange.json").write_text(json.dumps(
        {"model": meta["model"], "usage": meta["usage"], "prompt": prompt, "response": text,
         "code": match.group(1), "policy_path": f"results/policies/{version}.py"}, indent=1))
    print(text)


if __name__ == "__main__":
    main(*sys.argv[1:])
