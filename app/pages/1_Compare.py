"""Compare policies on the committed trials: success against one varied parameter, a clickable
map of every trial with its clip and reason, the TCP trajectory, and a trajectory from an mcap bag."""
from __future__ import annotations

import html
import sys
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from analysis import field as FD  # noqa: E402
from analysis import trials as T  # noqa: E402

st.set_page_config(page_title="GapHunter · Compare", layout="wide", initial_sidebar_state="collapsed")
st.markdown("<style>.block-container {padding-top: 1rem; max-width: 1280px;} video {max-height: 260px;}"
            "p.gh-cap {color: #8b8f98; font-size: .85rem;}</style>", unsafe_allow_html=True)
alt.theme.enable("dark")
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
OUTCOME_COLOURS = {"success": "#1baf7a", "miss": "#e34948", "collision": "#eda100", "timeout": "#6250d6",
                   "drop": "#e87ba4", "wrong_bin": "#eb6834", "failure": "#898781"}


@st.cache_data
def _trials(stamp: float) -> pd.DataFrame:
    return T.load_trials()


def _stamp() -> float:
    return max((p.stat().st_mtime for p in T.RESULTS.glob("*.csv")), default=0.0)


@st.cache_data
def _transcript(rel: str, mtime: float) -> dict | None:
    return T.read_transcript(rel)


@st.cache_data
def _mcap(path: str, mtime: float, every: int) -> pd.DataFrame:
    return T.mcap_trajectory(path, every=every)


df_all = _trials(_stamp())
st.title("Compare")
st.caption("Every number here comes from the committed csv files in results/: pass-1 rows, api_error rows dropped. "
           "Nothing is re-run.")

tasks = [t for t in ("peg-in-hole", "bin-sorting", "cube-in-bowl", "sphere-in-basket", "orange-in-bowl") if t in set(df_all["task"])]
task = st.radio("Task", tasks, horizontal=True)
df_task = df_all[df_all["task"] == task]
counts = df_task.groupby("policy").size()
policy_ids = sorted(counts.index, key=lambda p: (-counts[p], p))
default = [p for p in policy_ids if counts[p] >= 30][:4] or policy_ids[:4]
policies = st.multiselect("Policies", policy_ids, default=default, format_func=lambda p: f"{p} (n = {counts[p]})")
sets = st.multiselect("Trial sets", sorted(set(df_task["set"])), default=sorted(set(df_task["set"])))
df = df_task[df_task["policy"].isin(policies) & df_task["set"].isin(sets)]
colour = alt.Color("policy:N", scale=alt.Scale(domain=policies, range=PALETTE[:len(policies)]), legend=alt.Legend(title="policy"))

# ------------------------------------------------------------------ 1. success against one parameter
st.header("Success against one parameter")
st.caption("One parameter varies along the x axis. The other two are held near a value with the sliders: a trial "
           "counts only when both lie within the tolerance. Bands are Wilson 95 % intervals; hover for n.")
available = [p for p in T.PARAMS if p in df_task and df_task[p].notna().any()]
c1, c2, c3, c4 = st.columns([1.2, 1, 1, 1])
param = c1.radio("Vary", available, format_func=lambda p: T.PARAMS[p], horizontal=True)
others = [p for p in T.PLACEMENT if p != param and p in available][:2]
held = {}
for col, p in zip((c2, c3), others):
    lo, hi = float(df_task[p].min()), float(df_task[p].max())
    span = hi - lo
    centre = col.slider(f"hold {p} near", lo, hi, 0.0 if lo <= 0.0 <= hi else (lo + hi) / 2, step=span / 100 or 1.0)
    tol = col.slider(f"± tolerance on {p}", 0.0, span, span, step=span / 100 or 1.0)
    held[p] = (centre, tol)
bins = c4.slider("bins", 3, 12, 6)
mask = pd.Series(True, index=df.index)
for p, (centre, tol) in held.items():
    mask &= (df[p].astype(float) - centre).abs() <= tol
sliced = df[mask]
if param == "pose_error_mm":
    sliced = sliced[sliced["pose_error_mm"].notna()]
rates = T.binned_rate(sliced, param, bins)
if rates.empty:
    st.info("No trials fall inside this slice. Widen a tolerance or pick more trial sets.")
else:
    base = alt.Chart(rates).encode(x=alt.X("x:Q", title=T.PARAMS[param]), color=colour)
    band = base.mark_area(opacity=0.18).encode(y=alt.Y("lo:Q", title="success rate", scale=alt.Scale(domain=[0, 1])), y2="hi:Q")
    line = base.mark_line(point=True).encode(y="rate:Q", tooltip=["policy", alt.Tooltip("x:Q", format=".3f"), "n", "k",
                                                                   alt.Tooltip("rate:Q", format=".0%")])
    st.altair_chart((band + line).properties(height=320), width="stretch")
    st.caption(f"{len(sliced)} trials in the slice: " + ", ".join(
        f"{p} {int(n)}" for p, n in sliced.groupby('policy').size().items()))

# ------------------------------------------------------------------ 2. the trials map
st.header("Every trial")
st.caption("Each dot is one trial placed at the two parameters you choose, such as the sphere offset or the basket "
           "offset. Click a dot to see its clip, why it ended as it did, and "
           "the tool path. A clip that was never rendered is replayed from the transcript on the spot.")
pts = df.assign(seed=df["seed"].astype(int)).reset_index(drop=True)
axes = [p for p in T.PLACEMENT if p in available]
a1, a2 = st.columns(2)
x_param = a1.selectbox("x axis", axes, index=0, format_func=lambda p: T.PARAMS[p])
y_param = a2.selectbox("y axis", axes, index=min(1, len(axes) - 1), format_func=lambda p: T.PARAMS[p])
outcomes = [o for o in OUTCOME_COLOURS if o in set(pts["outcome"])]
scatter = (alt.Chart(pts).mark_point(size=60, filled=True, opacity=0.85)
           .encode(x=alt.X(f"{x_param}:Q", title=T.PARAMS[x_param]), y=alt.Y(f"{y_param}:Q", title=T.PARAMS[y_param]),
                   color=alt.Color("outcome:N", scale=alt.Scale(domain=outcomes, range=[OUTCOME_COLOURS[o] for o in outcomes])),
                   shape=alt.Shape("policy:N", scale=alt.Scale(domain=policies)),
                   tooltip=["policy", "set", "outcome", "seed", alt.Tooltip("dx:Q", format=".3f"),
                            alt.Tooltip("dy:Q", format=".3f"), alt.Tooltip("yaw:Q", format=".1f"),
                            alt.Tooltip("pose_error_mm:Q", format=".2f")])
           .add_params(alt.selection_point(name="pick", on="click", fields=["policy", "seed"]))
           .properties(height=420))
event = st.altair_chart(scatter, width="stretch", on_select="rerun", selection_mode="pick", key="trial_map")
picked = (event.selection.get("pick") or []) if event and hasattr(event, "selection") else []
row = None
if picked:
    sel = picked[0]
    hit = pts[(pts["policy"] == sel.get("policy")) & (pts["seed"] == int(sel.get("seed", -1)))]
    row = hit.iloc[0] if len(hit) else None
if row is None:
    st.info("Click a dot above to open that trial.")
else:
    st.subheader(f"{row['policy']} · seed {row['seed']} · {row['outcome']}")
    has_t = isinstance(row["transcript"], str) and (ROOT / str(row["transcript"])).is_file()
    t = _transcript(str(row["transcript"]), (ROOT / str(row["transcript"])).stat().st_mtime) if has_t else None
    v1, v2 = st.columns([1, 1.3], gap="medium")
    with v1:
        clip = T.clip_path(str(row["transcript"]) if has_t else "", render=has_t, row_clip=row.get("clip"))
        if clip:
            st.video(str(clip))
        else:
            st.info("No clip for this trial and no MuJoCo transcript to replay it from.")
        st.markdown(f'<p class="gh-cap">dx {row["dx"]:+.3f} m · dy {row["dy"]:+.3f} m · yaw {row["yaw"]:+.1f}° · '
                    f'{row["set"]} set · {row["csv"]}</p>', unsafe_allow_html=True)
    with v2:
        if t:
            st.markdown("**Why it ended** — " + html.escape(T.reason(t)))
            reply = T.last_reply(t)
            if reply:
                with st.expander("The model's last reply"):
                    st.code(reply, language=None, wrap_lines=True)
            traj = T.trajectory(t) if T.is_mujoco(t) else T.ditto_trajectory(str(row["transcript"]))
            if len(traj):
                nom = T.NOMINAL.get(row["sim"])
                target = pd.DataFrame([{"x": nom[0] + row["dx"], "y": nom[1] + row["dy"], "what": "true target"}]) if nom else None
                path = alt.Chart(traj).mark_line(point=True, color="#2a78d6").encode(
                    x=alt.X("x:Q", title="x (m)", scale=alt.Scale(zero=False)), y=alt.Y("y:Q", title="y (m)", scale=alt.Scale(zero=False)),
                    order="i:Q", tooltip=["i", "api", alt.Tooltip("z:Q", format=".4f"), "force_N"])
                if target is not None:
                    path = path + alt.Chart(target).mark_point(shape="cross", size=200, color="#e34948", filled=True).encode(x="x:Q", y="y:Q", tooltip="what")
                height = alt.Chart(traj).mark_line(color="#eb6834").encode(x=alt.X("i:Q", title="API call"), y=alt.Y("z:Q", title="z (m)"),
                                                                               tooltip=["i", "api", alt.Tooltip("z:Q", format=".4f"), "force_N"])
                st.altair_chart(alt.hconcat(path.properties(height=230, width=260, title="tool path, top view"),
                                            height.properties(height=230, width=260, title="tool height")), width="stretch")
        else:
            st.caption("No transcript for this trial.")

# ------------------------------------------------------------------ 2b. success-probability field
st.header("Success-probability field")
st.caption("Every trial is one draw from a coin whose bias depends on the placement. The field estimates that bias "
           "over the two axes of the map above for one policy, with the five estimators of the flow page. Masked "
           "cells have too little data behind them. The score is a 10-fold cross-validated Brier score, lower is better.")
f1, f2, f3 = st.columns([1.2, 1, 1])
field_policy = f1.selectbox("Field for", policies) if policies else None
est = f2.selectbox("Estimator", list(FD.ESTIMATORS), format_func=lambda e: FD.ESTIMATORS[e][0])
knob_spec = FD.ESTIMATORS[est][1]
knob = f3.slider(knob_spec[0], float(knob_spec[1]), float(knob_spec[2]), float(knob_spec[3])) if knob_spec else None
fpts = pts[(pts["policy"] == field_policy)] if field_policy else pts.iloc[0:0]
fpts = fpts[fpts[x_param].notna() & fpts[y_param].notna()]
if len(fpts) < FD.FIELD_MIN:
    st.info(f"At least {FD.FIELD_MIN} trials are needed for a field; {field_policy} has {len(fpts)} on these axes.")
else:
    box = ((float(fpts[x_param].min()), float(fpts[x_param].max())), (float(fpts[y_param].min()), float(fpts[y_param].max())))
    arr = fpts[[x_param, y_param, "success"]].astype(float).to_numpy()
    fit = FD.fitter(est, knob, box)
    table, note, cw, ch = FD.grid_field(fit, arr, box, g=32)
    score, scored = FD.cv_brier(fit, arr)
    table = table[table["support"] & table["p"].notna()]
    cells = alt.Chart(table).mark_rect().encode(
        x=alt.X("x:Q", title=T.PARAMS[x_param], bin=alt.Bin(step=cw)), y=alt.Y("y:Q", title=T.PARAMS[y_param], bin=alt.Bin(step=ch)),
        color=alt.Color("p:Q", title="P(success)", scale=alt.Scale(scheme="blues", domain=[0, 1])),
        tooltip=[alt.Tooltip("x:Q", format=".3f"), alt.Tooltip("y:Q", format=".3f"), alt.Tooltip("p:Q", format=".2f"),
                 alt.Tooltip("lo:Q", format=".2f"), alt.Tooltip("hi:Q", format=".2f")])
    dots = alt.Chart(fpts).mark_point(size=40, filled=True, stroke="#0b0b0b", strokeWidth=0.5).encode(
        x=f"{x_param}:Q", y=f"{y_param}:Q",
        color=alt.Color("outcome:N", scale=alt.Scale(domain=outcomes, range=[OUTCOME_COLOURS[o] for o in outcomes]), legend=None),
        tooltip=["seed", "outcome"])
    st.altair_chart((cells + dots).properties(height=420), width="stretch")
    st.caption(f"{field_policy}: {len(fpts)} trials · {note} · cross-validated Brier {score:.3f} on {scored} trials")

# ------------------------------------------------------------------ 3. mcap replay
st.header("Trajectory from an mcap bag")
st.caption("For the Gazebo trials: give the path of a ROS 2 mcap bag and the tool path is read from "
           "/aic_controller/controller_state.")
mcap_path = st.text_input("mcap path", value="", placeholder="/path/to/bag_trial_1_..._0.mcap")
if mcap_path:
    p = Path(mcap_path).expanduser()
    if not p.is_file():
        st.warning("That file does not exist.")
    else:
        try:
            tr = _mcap(str(p), p.stat().st_mtime, 10)
        except Exception as exc:  # noqa: BLE001 - a bag without the topic or the package is a user-facing message
            st.error(f"Could not read the bag: {exc}")
            tr = pd.DataFrame()
        if len(tr):
            st.caption(f"{len(tr)} samples over {tr['t_s'].iloc[-1]:.1f} s (every 10th message)")
            top = alt.Chart(tr).mark_line(color="#2a78d6").encode(x=alt.X("x:Q", scale=alt.Scale(zero=False)), y=alt.Y("y:Q", scale=alt.Scale(zero=False)), order="t_s:Q", tooltip=[alt.Tooltip("t_s:Q", format=".1f"), "x", "y", "z"])
            z = alt.Chart(tr).mark_line(color="#eb6834").encode(x=alt.X("t_s:Q", title="time (s)"), y=alt.Y("z:Q", title="z (m)"), tooltip=[alt.Tooltip("t_s:Q", format=".1f"), "z"])
            st.altair_chart(alt.hconcat(top.properties(height=280, width=420, title="tool path, top view"), z.properties(height=280, width=420, title="tool height")), width="stretch")
