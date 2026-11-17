"""Compare policies on every committed trial: success against one varied parameter, a clickable map of
every trial with its clip, and the success-probability field with five estimators."""
from __future__ import annotations

import html

import altair as alt
import pandas as pd
import streamlit as st

import field as FD
import trials as T

st.set_page_config(page_title="Matrix Lab · Compare", layout="wide", initial_sidebar_state="collapsed")
st.markdown("<style>.block-container {padding-top: 1rem; max-width: 1280px;} video {max-height: 260px;}"
            "p.gh-cap {color: #8b8f98; font-size: .85rem;}</style>", unsafe_allow_html=True)
alt.theme.enable("dark")
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
OUTCOME_COLOURS = {"success": "#1baf7a", "miss": "#e34948", "collision": "#eda100", "timeout": "#6250d6",
                   "drop": "#e87ba4", "wrong_bin": "#eb6834", "failure": "#898781"}


@st.cache_data(show_spinner="Loading the trials from the dataset…")
def _trials() -> pd.DataFrame:
    return T.load_trials()


df_all = _trials()
st.title("Compare")
st.caption("Every number here comes from the trial csv files of the Matrix-Lab/Matrix-Lab-Preliminary-Experiments "
           "dataset: pass-1 rows, api_error rows dropped. Nothing is re-run.")

tasks = [t for t in ("peg-in-hole", "bin-sorting", "bowl-sorting", "sphere-in-basket", "orange-in-bowl") if t in set(df_all["task"])]
task = st.radio("Task", tasks, horizontal=True)
df_task = df_all[df_all["task"] == task]
counts = df_task.groupby("policy").size()
policy_ids = sorted(counts.index, key=lambda p: (-counts[p], p))
default = [p for p in policy_ids if counts[p] >= 30][:4] or policy_ids[:4]
policies = st.multiselect("Policies", policy_ids, default=default, format_func=lambda p: f"{p} (n = {counts[p]})")
sets = st.multiselect("Trial sets", sorted(set(df_task["set"])), default=sorted(set(df_task["set"])))
df = df_task[df_task["policy"].isin(policies) & df_task["set"].isin(sets)]
colour = alt.Color("policy:N", scale=alt.Scale(domain=policies, range=PALETTE[:len(policies)]), legend=alt.Legend(title="policy"))
available = [p for p in T.PARAMS if p in df_task and df_task[p].notna().any()]

# ------------------------------------------------------------------ 1. success against one parameter
st.header("Success against one parameter")
st.caption("One parameter varies along the x axis. The other placement parameters are held near a value with the "
           "sliders: a trial counts only when they lie within the tolerance. Bands are Wilson 95 % intervals; hover for n.")
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
rates = T.binned_rate(sliced, param, bins) if len(sliced) else pd.DataFrame()
if rates.empty:
    st.info("No trials fall inside this slice. Widen a tolerance or pick more trial sets.")
else:
    base = alt.Chart(rates).encode(x=alt.X("x:Q", title=T.PARAMS[param]), color=colour)
    band = base.mark_area(opacity=0.18).encode(y=alt.Y("lo:Q", title="success rate", scale=alt.Scale(domain=[0, 1])), y2="hi:Q")
    line = base.mark_line(point=True).encode(y="rate:Q", tooltip=["policy", alt.Tooltip("x:Q", format=".3f"), "n", "k", alt.Tooltip("rate:Q", format=".0%")])
    st.altair_chart((band + line).properties(height=320), width="stretch")
    st.caption(f"{len(sliced)} trials in the slice: " + ", ".join(f"{p} {int(n)}" for p, n in sliced.groupby("policy").size().items()))

# ------------------------------------------------------------------ 2. the trials map
st.header("Every trial")
st.caption("Each dot is one trial placed at the two parameters you choose. Click a dot to see its clip and how it ended.")
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
                   tooltip=["policy", "set", "outcome", "seed", alt.Tooltip("dx:Q", format=".3f"), alt.Tooltip("dy:Q", format=".3f"),
                            alt.Tooltip("yaw:Q", format=".1f"), alt.Tooltip("pose_error_mm:Q", format=".2f")])
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
    v1, v2 = st.columns([1, 1.3], gap="medium")
    with v1:
        if isinstance(row.get("clip"), str) and row["clip"]:
            st.video(row["clip"])
        else:
            st.info("No clip was recorded for this trial.")
        st.markdown(f'<p class="gh-cap">dx {row["dx"]:+.3f} m · dy {row["dy"]:+.3f} m · yaw {row["yaw"]:+.1f}° · '
                    f'{row["set"]} set · {row["csv"]}</p>', unsafe_allow_html=True)
    with v2:
        st.markdown("**How it ended** — " + html.escape(T.reason(row)))

# ------------------------------------------------------------------ 3. success-probability field
st.header("Success-probability field")
st.caption("Every trial is one draw from a coin whose bias depends on the placement. The field estimates that bias over "
           "the two axes of the map for one policy, with five estimators. Masked cells have too little data behind them. "
           "The score is a 10-fold cross-validated Brier score, lower is better.")
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
