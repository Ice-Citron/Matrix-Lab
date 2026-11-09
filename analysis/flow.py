"""Build docs/flow.html: a click-through product demo (Task -> Policies -> Sweep -> Run -> Results) for Pages.

    python -m analysis.flow       # run from a clean checkout of the commit to publish

Every number, row, clip and sentence comes from committed files: results/policies.json, the CSVs it points to,
results/featured.json, transcripts, runner/policies.py (local boxes), sim/tools/ditto_vla_batch.py (Ditto boxes)
and, for pi05, results/policies.json in commit ab7b5c7. The page replays the committed CSV rows; nothing is
simulated in the browser. Options this build cannot run are greyed out and say so.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.coverage import (POLICY_COLUMNS, SEARCH_RANGE_MM, _range, coverage_report, default_boxes,
                               first_pass, fold_box, inside_local, load_policies, policy_table)
from sim.ditto_trial import param_hash

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "flow.html"
ASSETS = ROOT / "docs" / "assets" / "flow"
BINS = 16  # the ±0.05 m local box spans 4×4 cells of the ±0.20 m deployment box
TASK_OF_SIM = {"mujoco": "peg", "ditto": "sphere", "sort": "sort"}


def _ditto_boxes() -> tuple[dict, dict]:
    src = (ROOT / "sim/tools/ditto_vla_batch.py").read_text()

    def get(name):  # the dict literal uses tuples; turn it into JSON
        lit = re.search(rf"^{name} = (\{{.*?\}})", src, re.M).group(1)
        return {k: tuple(v) for k, v in json.loads(lit.replace("(", "[").replace(")", "]")).items()}

    return get("DITTO_LOCAL"), get("DITTO_DEPLOY")


def _local_boxes() -> dict:
    try:
        from runner.policies import POLICIES
        return {m["policy"]: m["local_box"] for m in POLICIES if m.get("local_box")}
    except Exception:
        return {}


def _history_rows(names=("pi05", "gemini-robotics", "gr00t")) -> list[dict]:
    """Planned rows that were dropped from policies.json, from its committed version in ab7b5c7."""
    try:
        old = json.loads(subprocess.run(["git", "show", "ab7b5c7:results/policies.json"], cwd=ROOT,
                                        capture_output=True, text=True, check=True).stdout)
        return [r for r in old if r["policy"] in names]
    except Exception:
        return []


# Illustrative placeholders for the policies table: no checkpoint, no data, always greyed out.
PLACEHOLDERS = [
    {"policy": "act-finetune", "family": "Imitation learning (ACT)", "model": "your fine-tuned checkpoint",
     "sim": "mujoco", "notes": "Placeholder: an ACT policy fine-tuned on your demonstrations."},
    {"policy": "custom-upload", "family": "Custom", "model": "you",
     "sim": None, "notes": "Placeholder: any policy that implements run_trial(params) -> TrialResult."},
]


def _copy(rel: str) -> str | None:
    src = ROOT / rel
    if not rel or not src.is_file():
        return None
    (ASSETS / "clips").mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, ASSETS / "clips" / src.name)
    return f"assets/flow/clips/{src.name}"


def _headline(feat: dict) -> str:
    t = ROOT / str(feat.get("failure_transcript") or "")
    if not feat.get("headline") or not t.is_file():
        return ""
    d = json.loads(t.read_text())
    o = d["outcome"]
    return feat["headline"].format(**d["params"], **o, search_mm=SEARCH_RANGE_MM,
                                   ratio=o["pose_error_mm"] / SEARCH_RANGE_MM)


def _box_json(b: dict) -> dict:
    return {k: [round(v, 4) for v in _range(b, k)] for k in ("dx", "dy", "yaw")}


def _criteria() -> dict:
    """Scoring thresholds for the trial popup, read from the simulator sources so they cannot drift."""
    src = (ROOT / "sim/mj_sim.py").read_text()
    num = lambda name: float(re.search(rf"^{name} = ([\d.]+)", src, re.M).group(1))  # noqa: E731
    return {"peg": {"depth_mm": num("SUCCESS_DEPTH") * 1000, "lateral_mm": num("SUCCESS_LATERAL") * 1000,
                    "force_N": num("COLLISION_N"), "api_calls": int(num("MAX_API_CALLS"))},
            "sort": {"force_N": num("COLLISION_N"), "api_calls": int(num("MAX_API_CALLS"))},  # mj_sort_sim imports both
            "sphere": {"lift_progress": 0.3}}


# Each peg-in-hole policy's lateral search around its hole estimate, in mm.
SEARCH_MM = {
    "astra-script": SEARCH_RANGE_MM,  # results/policies/v1.py: 0.6 mm grid, 3 rings
    "opus-script": 0.8,  # results/policies/opus_v1.py: 0.4 mm grid, r <= 0.8 mm
}


def _detail(r: pd.Series) -> dict | None:
    """Per-trial scoring detail for the page's popup: CSV extras plus the transcript's outcome, when there is one.
    Keys are short because the page carries one of these per trial: s seed, n decisions, w wall seconds,
    t transcript path under results/ without .json; MuJoCo d depth mm, l lateral mm, y yaw error deg,
    f peak force N, e error; bin sorting c cube colour, b bin it ended in, lf lifted; Ditto pk peak progress,
    pg final progress, st control steps, ss sim seconds, tr termination reason."""
    d = {}
    for k, kk in (("seed", "s"), ("n_decisions", "n"), ("t_wall", "w")):
        v = r.get(k)
        if v is not None and not (isinstance(v, float) and np.isnan(v)):
            d[kk] = round(float(v), 1) if k == "t_wall" else int(v)
    t = r.get("transcript")
    if isinstance(t, str) and t and (ROOT / t).is_file():
        d["t"] = t.removeprefix("results/").removesuffix(".json")
        j = json.loads((ROOT / t).read_text())
        o = j.get("outcome") if isinstance(j, dict) else None
        if isinstance(o, dict):  # MuJoCo peg-in-hole: sim/mj_sim.py outcome()
            for k, kk in (("depth_mm", "d"), ("lateral_mm", "l"), ("yaw_err_deg", "y"), ("max_force_N", "f")):
                if o.get(k) is not None:
                    d[kk] = round(float(o[k]), 2)
            if o.get("colour"):  # bin sorting: sim/mj_sort_sim.py outcome()
                d["c"], d["b"], d["lf"] = o["colour"], o.get("bin") or "", int(bool(o.get("lifted")))
            if o.get("error"):
                d["e"] = str(o["error"])[:160]
        elif isinstance(j, dict) and "oracle_success" in j:  # Ditto / ManiSkill evaluator row
            for k, kk in (("progress_score", "pk"), ("final_progress_score", "pg"), ("steps", "st"),
                          ("sim_time_s", "ss")):
                if j.get(k) is not None:
                    d[kk] = round(float(j[k]), 2)
            if j.get("termination_reason"):
                d["tr"] = str(j["termination_reason"])
    return d or None


def _policy(pol: dict, lb: dict, db: dict, feat: dict) -> dict:
    pid, sim = pol["policy"], pol.get("sim") or "mujoco"
    frames = []
    for phase, key in ((0, "local_csv"), (1, "deploy_csv")):
        if pol.get(key) and (ROOT / pol[key]).is_file():
            frames.append(first_pass(pd.read_csv(ROOT / pol[key])).assign(_phase=phase))
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    xlo, xhi = _range(db, "dx")
    ylo, yhi = _range(db, "dy")
    rows, clips = [], []
    if pid == "astra-script" and (url := _copy(feat.get("success_clip", ""))):
        clips.append({"url": url, "mode": "success", "dx": 0.0, "dy": 0.0, "yaw": 0, "phase": "nominal"})
    if len(df):
        df[["dx", "dy", "yaw"]] = df[["dx", "dy", "yaw"]].astype(float)
        inside = inside_local(df, lb).to_numpy()
        ix = np.clip(np.digitize(df["dx"], np.linspace(xlo, xhi, BINS + 1)) - 1, 0, BINS - 1)
        iy = np.clip(np.digitize(df["dy"], np.linspace(ylo, yhi, BINS + 1)) - 1, 0, BINS - 1)
        succ = df["success"].astype(str).str.lower().isin(("true", "1")).to_numpy()
        for i, (_, r) in enumerate(df.iterrows()):
            fm = r.get("failure_mode")
            mode = "" if succ[i] else (str(fm) if isinstance(fm, str) and fm else "failure")
            c = -1
            clip = r.get("clip") if isinstance(r.get("clip"), str) else None
            if clip is None and sim == "mujoco":  # clip rendered after the sweep: named by param hash
                hashed = f"results/clips/{param_hash(r)}.mp4"
                clip = hashed if (ROOT / hashed).is_file() else None
            if clip and (url := _copy(clip)):
                c = len(clips)
                clips.append({"url": url, "mode": mode or "success", "dx": round(float(r["dx"]), 3),
                              "dy": round(float(r["dy"]), 3), "yaw": round(float(r["yaw"])),
                              "phase": "local" if r["_phase"] == 0 else "deployment"})
            pe = r.get("pose_error_mm")
            # [phase, success, mode, inside local, cell ix, cell iy, clip index, pose error mm, dx, dy, yaw, detail]
            rows.append([int(r["_phase"]), int(succ[i]), mode, int(inside[i]), int(ix[i]), int(iy[i]), c,
                         round(float(pe), 2) if pd.notna(pe) else None,
                         round(float(r["dx"]), 3), round(float(r["dy"]), 3), round(float(r["yaw"]), 1), _detail(r)])
    empty = pd.DataFrame(columns=["dx", "dy", "yaw", "success"])
    ldf = first_pass(pd.read_csv(ROOT / pol["local_csv"])) if pol.get("local_csv") else empty
    ddf = first_pass(pd.read_csv(ROOT / pol["deploy_csv"])) if pol.get("deploy_csv") else empty
    rep = coverage_report(ldf, ddf, lb, db)
    last = (f"{rep['outside_local_success']:.0%} outside local box" if rep["n_deploy"]
            else f"{rep['local_success']:.0%} local (n={rep['n_local']})" if rep["n_local"] else "")
    return {"id": pid, "family": pol.get("family") or "", "author": pol.get("model") or "", "sim": sim,
            "task": TASK_OF_SIM.get(sim, sim), "note": pol.get("notes") or "", "status": pol.get("status") or "",
            "last": last, "rows": rows, "clips": clips, "why": _headline(feat) if pid == "astra-script" else "",
            "search_mm": SEARCH_MM.get(pid),
            "box": {"local": _box_json(fold_box(lb)), "deploy": _box_json(db)},
            "check": {"local": rep["local_success"], "deploy": rep["deploy_success"],
                      "outside": rep["outside_local_success"], "n_outside": rep["n_outside"]}}


def build(readme: Path | None = None) -> Path:
    from runner.sample import sort_deploy_box
    lb0, db0 = default_boxes()
    dl, dd = _ditto_boxes()
    sd = sort_deploy_box()  # bin sorting reuses lb0 as its local box
    deploy_of = {"ditto": dd, "sort": sd}
    boxes = _local_boxes()
    feat = json.loads((ROOT / "results/featured.json").read_text())
    if ASSETS.exists():
        shutil.rmtree(ASSETS)
    ASSETS.mkdir(parents=True)
    pols = []
    for pol in load_policies("results/policies.json", root=str(ROOT)):
        sim = pol.get("sim") or "mujoco"
        pols.append(_policy(pol, boxes.get(pol["policy"], dl if sim == "ditto" else lb0),
                            dd if sim == "ditto" else db0, feat))
    # Same policy run on another task: committed results/<sim>_<policy>_{local,deploy}.csv that policies.json
    # does not reference (it holds one row per policy name). Shown as "<policy> (<task>)".
    listed = {pol.get(k) for pol in load_policies("results/policies.json", root=str(ROOT))
              for k in ("local_csv", "deploy_csv")}
    tracked = set(subprocess.run(["git", "ls-files", "results/*.csv"], cwd=ROOT, capture_output=True,
                                 text=True).stdout.split())
    for base in list(load_policies("results/policies.json", root=str(ROOT))):
        for sim, label in (("ditto", "sphere"), ("mujoco", "peg"), ("sort", "sorting")):
            if sim == (base.get("sim") or "mujoco"):
                continue
            csvs = {k: f"results/{sim}_{base['policy']}_{k.split('_')[0]}.csv" for k in ("local_csv", "deploy_csv")}
            csvs = {k: v for k, v in csvs.items() if v in tracked and v not in listed}
            if not csvs:
                continue
            short = base["policy"].split("-")[0]
            src = next((f for f in (f"results/policies/{sim}_{short}_v1.py",) if f in subprocess.run(
                ["git", "ls-files", "results/policies"], cwd=ROOT, capture_output=True, text=True).stdout.split()), "")
            n_deploy = len(first_pass(pd.read_csv(ROOT / csvs["deploy_csv"]))) if "deploy_csv" in csvs else 0
            alt = {**base, **csvs, "policy": f"{base['policy']} ({label})", "sim": sim,
                   "status": "full" if n_deploy >= 600 else "smoke",  # 600 = the full peg deploy sweep
                   "notes": f"{base.get('model')} wrote this policy once for the {label} task"
                            + (f" ({src})" if src else "") + "; it runs unchanged on every trial."}
            pols.append(_policy(alt, dl if sim == "ditto" else lb0, deploy_of.get(sim, db0), feat))
    # Requested for filming: full peg-in-hole runs also listed on the sphere task, clearly labelled as
    # peg-in-hole data (their own boxes, rows and clips). They are never presented as sphere-task results.
    # rows and clips are not repeated in the JSON; the page resolves "same" to the original policy's arrays.
    for p in [p for p in pols if p["task"] == "peg" and p["status"] == "full"]:
        pols.append({**p, "id": f"{p['id']} (peg-in-hole data)", "task": "sphere", "borrowed": True,
                     "rows": [], "clips": [], "same": p["id"],
                     "note": "Peg-in-hole (MuJoCo) results shown on this task for comparison only. "
                             "Not sphere-task results.", "why": ""})
    for old in _history_rows() + PLACEHOLDERS:
        if any(p["id"] == old["policy"] for p in pols):
            continue
        sim = old.get("sim")
        pols.append({"id": old["policy"], "family": old.get("family") or "", "author": old.get("model") or "",
                     "sim": sim or "", "task": TASK_OF_SIM.get(sim) if sim else "", "note": old.get("notes") or "",
                     "status": "planned", "last": "", "rows": [], "clips": [], "why": "", "box": None,
                     "check": None, "placeholder": old in PLACEHOLDERS})
    table = policy_table("results/policies.json", root=str(ROOT), local_box=lb0, deploy_box=db0)
    shutil.copyfile(readme or ROOT / "README.md", ASSETS / "README.md")
    data = {"policies": pols, "bins": BINS, "criteria": _criteria(),
            "tasks": {"peg": {"local": _box_json(fold_box(lb0)), "deploy": _box_json(db0)},
                      "sphere": {"local": _box_json(dl), "deploy": _box_json(dd)},
                      "sort": {"local": _box_json(fold_box(lb0)), "deploy": _box_json(sd)}},
            "table": {"cols": POLICY_COLUMNS, "rows": [[str(r[c]) for c in POLICY_COLUMNS]
                                                         for _, r in table.iterrows()]}}
    OUT.write_text(PAGE.replace("/*DATA*/", json.dumps(data, separators=(",", ":"))))
    return OUT


PAGE = (Path(__file__).with_name("flow_page.html")).read_text() if (Path(__file__).with_name("flow_page.html")).exists() else ""

if __name__ == "__main__":
    print(build())
