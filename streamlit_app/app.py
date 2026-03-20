"""
DIAGONAL: Human Evaluation & Results Dashboard
================================================
Streamlit Community Cloud compatible version.
Launch: streamlit run streamlit_app/app.py
"""

import streamlit as st
import json
import random
import datetime
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from collections import defaultdict

# ---------------------------------------------------------------------------
# Paths (relative to repo root)
# ---------------------------------------------------------------------------
APP_DIR = Path(__file__).parent
DATA_DIR = APP_DIR / "data"
PROMPTS_PATH = DATA_DIR / "prompts_subset.json"
EVAL_RESULTS_PATH = DATA_DIR / "vlm_fullscale_merged.json"
KEYFRAMES_DIR = DATA_DIR / "keyframes"
HUMAN_EVAL_OUTPUT = DATA_DIR / "human_eval_results.json"

MODELS = ["storydiff", "echoshot", "vgot", "vic"]
MODEL_LABELS = {
    "storydiff": "StoryDiffusion",
    "echoshot": "EchoShot",
    "vgot": "VGoT",
    "vic": "VIC",
}
NUM_SCENARIOS = 40
SEED = 42

PATTERN_DESC = {
    "Relay": "A \u2192 AB \u2192 B",
    "Sequential_Relay": "A \u2192 AB \u2192 B",
    "Split": "AB \u2192 A \u2192 B",
    "Accumulation": "A \u2192 AB \u2192 ABC",
    "Convergence": "A \u2192 B \u2192 AB",
    "Sliding_Window": "AB \u2192 BC \u2192 C",
    "Reduction": "ABC \u2192 AB \u2192 A",
    "Reverse_Relay": "B \u2192 AB \u2192 A",
}

LIKERT_LABELS = {
    1: "1 - No compliance",
    2: "2 - Poor",
    3: "3 - Moderate",
    4: "4 - Good",
    5: "5 - Perfect compliance",
}

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

@st.cache_data
def load_eval_results():
    with open(EVAL_RESULTS_PATH) as f:
        return json.load(f)

@st.cache_data
def load_prompts():
    with open(PROMPTS_PATH) as f:
        return json.load(f)

@st.cache_data
def build_eval_index():
    data = load_eval_results()
    idx = {}
    by_prefix = {}
    for entry in data:
        idx[(entry["method"], entry["vid"])] = entry
        prefix = "_".join(entry["vid"].split("_")[:2])
        by_prefix.setdefault(entry["method"], {})[prefix] = entry
    return idx, by_prefix

# ---------------------------------------------------------------------------
# Page 1: Results Dashboard
# ---------------------------------------------------------------------------

def page_dashboard():
    st.header("STEM Evaluation Results Dashboard")
    data = load_eval_results()
    df = pd.DataFrame(data)

    # Overall metrics
    st.subheader("Overall PTM & EPA by Model")
    summary = df.groupby("method").agg(
        PTM_mean=("match_prescribed", "mean"),
        PTM_std=("match_prescribed", "std"),
        EPA_mean=("epa", "mean"),
        EPA_std=("epa", "std"),
        n=("vid", "count"),
    ).reindex(MODELS)
    summary.index = [MODEL_LABELS.get(m, m) for m in summary.index]
    st.dataframe(summary.round(3), use_container_width=True)

    # Bar chart
    fig = go.Figure()
    for metric, color in [("match_prescribed", "#4285F4"), ("epa", "#34A853")]:
        means = df.groupby("method")[metric].mean().reindex(MODELS)
        stds = df.groupby("method")[metric].std().reindex(MODELS)
        labels = [MODEL_LABELS[m] for m in MODELS]
        name = "PTM" if "match" in metric else "EPA"
        fig.add_trace(go.Bar(
            x=labels, y=means, error_y=dict(type="data", array=stds, visible=True),
            name=name, marker_color=color,
        ))
    fig.update_layout(barmode="group", yaxis_title="Score", height=400)
    st.plotly_chart(fig, use_container_width=True)

    # Per-pattern heatmap
    st.subheader("Per-Pattern PTM Heatmap")
    pivot = df.pivot_table(values="match_prescribed", index="pattern", columns="method", aggfunc="mean")
    pivot = pivot[MODELS]
    pivot.columns = [MODEL_LABELS[m] for m in MODELS]
    fig2 = px.imshow(
        pivot.round(3), text_auto=True, color_continuous_scale="Blues",
        labels=dict(color="PTM"), aspect="auto",
    )
    fig2.update_layout(height=350)
    st.plotly_chart(fig2, use_container_width=True)

    # Failure analysis
    st.subheader("Failure Type Distribution")
    failure_data = []
    for d in data:
        m = d["method"]
        E = d["E_error"]
        dS = d["delta_S_ideal"]
        types = set()
        for t in range(len(E)):
            for e in range(len(E[t])):
                if abs(E[t][e]) == 2:
                    types.add("IV: Reversal")
                elif dS[t][e] == -1 and E[t][e] > 0:
                    types.add("I: Context Bleeding")
                elif dS[t][e] == 1 and E[t][e] < 0:
                    types.add("II: Amnesia")
                elif dS[t][e] == 0 and E[t][e] != 0:
                    types.add("III: Spurious")
        if not types:
            types.add("None (perfect)")
        for ft in types:
            failure_data.append({"Model": MODEL_LABELS[m], "Failure Type": ft})

    fdf = pd.DataFrame(failure_data)
    total_per_model = fdf.groupby("Model").size()
    fdf_pct = fdf.groupby(["Model", "Failure Type"]).size().reset_index(name="count")
    fdf_pct["pct"] = fdf_pct.apply(
        lambda r: 100 * r["count"] / len([d for d in data if MODEL_LABELS[d["method"]] == r["Model"]]), axis=1
    )
    fig3 = px.bar(fdf_pct, x="Model", y="pct", color="Failure Type", barmode="group",
                  labels={"pct": "% of videos"}, height=400)
    st.plotly_chart(fig3, use_container_width=True)

    # Coherence-Compliance trade-off
    st.subheader("Coherence-Compliance Trade-off")
    ptm_means = df.groupby("method")["match_prescribed"].mean().reindex(MODELS)
    ic_values = {"storydiff": 0.81, "echoshot": 0.85, "vgot": 0.88, "vic": 0.94}
    fig4 = go.Figure()
    for m in MODELS:
        fig4.add_trace(go.Scatter(
            x=[ptm_means[m]], y=[ic_values[m]],
            mode="markers+text", text=[MODEL_LABELS[m]],
            textposition="top center", marker=dict(size=15),
            name=MODEL_LABELS[m],
        ))
    fig4.update_layout(
        xaxis_title="PTM (Narrative Compliance) \u2192",
        yaxis_title="Identity Consistency \u2192",
        height=400, showlegend=False,
    )
    fig4.add_annotation(
        text="\u03c1\u209b = -1.0 (n=4)", x=0.15, y=0.92, showarrow=False,
        font=dict(size=14, color="red"),
    )
    st.plotly_chart(fig4, use_container_width=True)


# ---------------------------------------------------------------------------
# Page 2: Human Evaluation
# ---------------------------------------------------------------------------

def page_human_eval():
    st.header("Human Evaluation: Narrative Compliance")

    evaluator = st.sidebar.text_input("Your name", key="eval_name")
    if not evaluator:
        st.warning("Please enter your name in the sidebar to begin.")
        return

    prompts = load_prompts()
    eval_idx, by_prefix = build_eval_index()

    prompt_keys = list(prompts.keys())
    rng = random.Random(SEED)
    scenarios = rng.sample(prompt_keys, min(NUM_SCENARIOS, len(prompt_keys)))

    # Assign model pairs
    rng2 = random.Random(SEED)
    model_pairs = {}
    for sid in scenarios:
        ms = MODELS[:]
        rng2.shuffle(ms)
        pair = ms[:2]
        if rng2.random() < 0.5:
            pair = pair[::-1]
        model_pairs[sid] = {"A": pair[0], "B": pair[1]}

    # Session state
    if "eval_idx" not in st.session_state:
        st.session_state.eval_idx = 0

    idx = st.session_state.eval_idx
    st.sidebar.metric("Progress", f"{idx}/{len(scenarios)}")
    st.sidebar.progress(idx / max(len(scenarios), 1))

    if idx >= len(scenarios):
        st.success("All evaluations complete! Thank you.")
        st.balloons()
        return

    sid = scenarios[idx]
    scenario = prompts[sid]
    meta = scenario["metadata"]
    pattern = meta["pattern_type"]
    pair = model_pairs[sid]
    prefix = "_".join(sid.split("_")[:2])

    # Header
    st.subheader(f"Scenario {idx+1}/{len(scenarios)}")
    desc = PATTERN_DESC.get(pattern, pattern)
    st.markdown(f"**Pattern:** {pattern.replace('Sequential_', '')} ({desc})")
    entities = list(meta["core_entities"].values())
    st.markdown(f"**Entities:** {', '.join(entities)}")

    # Prescribed presence
    for m_name in [pair["A"], pair["B"]]:
        entry = by_prefix.get(m_name, {}).get(prefix)
        if entry and "S_ideal" in entry:
            s_ideal = entry["S_ideal"]
            cols_header = entities[:len(s_ideal[0])] if s_ideal else entities
            df_s = pd.DataFrame(
                s_ideal,
                columns=cols_header[:len(s_ideal[0])],
                index=[f"Shot {i+1}" for i in range(len(s_ideal))],
            )
            st.markdown("**Prescribed Presence S*:**")
            st.dataframe(df_s.replace({1: "\u2705", 0: "\u274c"}), use_container_width=True)
            break

    # Show keyframes side by side
    st.markdown("---")
    col_a, col_b = st.columns(2)
    col_a.markdown("### Video A")
    col_b.markdown("### Video B")

    for col, side in [(col_a, "A"), (col_b, "B")]:
        model = pair[side]
        # Find vid for this model
        entry = by_prefix.get(model, {}).get(prefix)
        if entry:
            vid = entry["vid"]
            if model == "storydiff":
                kf_dir = KEYFRAMES_DIR / vid
                for i in range(1, 4):
                    img_path = kf_dir / f"shot0{i}.jpg"
                    if not img_path.exists():
                        img_path = kf_dir / f"shot0{i}.png"
                    if img_path.exists():
                        col.image(str(img_path), caption=f"Shot {i}", use_container_width=True)
                    else:
                        col.info(f"Shot {i}: image not available on cloud")
            else:
                col.info(f"Keyframes for {MODEL_LABELS[model]} not available on cloud.\n\nPTM: {entry['match_prescribed']:.2f}, EPA: {entry['epa']:.2f}")
                # Show S_obs as alternative
                if "S_obs" in entry:
                    s_obs_df = pd.DataFrame(
                        entry["S_obs"],
                        columns=entry.get("entities", entities)[:len(entry["S_obs"][0])],
                        index=[f"Shot {i+1}" for i in range(len(entry["S_obs"]))],
                    )
                    col.markdown("**Observed S_obs:**")
                    col.dataframe(s_obs_df.replace({1: "\u2705", 0: "\u274c"}), use_container_width=True)

    # Rating
    st.markdown("---")
    c1, c2 = st.columns(2)
    with c1:
        likert_a = st.radio("Rate Video A:", [1,2,3,4,5],
                            format_func=lambda x: LIKERT_LABELS[x], key=f"la_{idx}", index=2)
    with c2:
        likert_b = st.radio("Rate Video B:", [1,2,3,4,5],
                            format_func=lambda x: LIKERT_LABELS[x], key=f"lb_{idx}", index=2)
    pref = st.radio("Which follows the narrative better?", ["A", "B", "Tie"],
                    key=f"pref_{idx}", horizontal=True)

    if st.button("Next", type="primary", use_container_width=True):
        record = {
            "evaluator": evaluator,
            "timestamp": datetime.datetime.now().isoformat(),
            "vid": sid,
            "pattern": pattern,
            "model_a": pair["A"],
            "model_b": pair["B"],
            "likert_a": likert_a,
            "likert_b": likert_b,
            "preference": pref,
        }
        results = []
        if HUMAN_EVAL_OUTPUT.exists():
            with open(HUMAN_EVAL_OUTPUT) as f:
                results = json.load(f)
        results.append(record)
        with open(HUMAN_EVAL_OUTPUT, "w") as f:
            json.dump(results, f, indent=2)
        st.session_state.eval_idx = idx + 1
        st.rerun()


# ---------------------------------------------------------------------------
# Page 3: Evaluation Results
# ---------------------------------------------------------------------------

def page_eval_results():
    st.header("Human Evaluation Results")
    if not HUMAN_EVAL_OUTPUT.exists():
        st.info("No evaluations submitted yet.")
        return
    with open(HUMAN_EVAL_OUTPUT) as f:
        results = json.load(f)
    st.metric("Total evaluations", len(results))
    if not results:
        return

    rdf = pd.DataFrame(results)

    # Per-model scores
    rows = []
    for _, r in rdf.iterrows():
        rows.append({"model": r["model_a"], "score": r["likert_a"]})
        rows.append({"model": r["model_b"], "score": r["likert_b"]})
    scores_df = pd.DataFrame(rows)
    st.subheader("Mean Likert by Model")
    summary = scores_df.groupby("model")["score"].agg(["mean", "std", "count"])
    summary.index = [MODEL_LABELS.get(m, m) for m in summary.index]
    st.dataframe(summary.round(2), use_container_width=True)

    # Preference
    st.subheader("Pairwise Preferences")
    wins = defaultdict(int)
    for _, r in rdf.iterrows():
        if r["preference"] == "A":
            wins[r["model_a"]] += 1
        elif r["preference"] == "B":
            wins[r["model_b"]] += 1
    wins_df = pd.DataFrame([
        {"Model": MODEL_LABELS.get(m, m), "Wins": wins[m]}
        for m in MODELS
    ])
    st.dataframe(wins_df, use_container_width=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    st.set_page_config(
        page_title="DIAGONAL - Narrative Evaluation",
        page_icon="\U0001f3ac",
        layout="wide",
    )
    st.sidebar.title("\U0001f3ac DIAGONAL")
    page = st.sidebar.radio(
        "Navigation",
        ["Dashboard", "Human Evaluation", "Eval Results"],
    )
    if page == "Dashboard":
        page_dashboard()
    elif page == "Human Evaluation":
        page_human_eval()
    else:
        page_eval_results()

if __name__ == "__main__":
    main()
