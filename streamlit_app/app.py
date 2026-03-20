"""
DIAGONAL: 내러티브 평가 대시보드 & 인간 평가
=============================================
streamlit run streamlit_app/app.py
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
# Paths
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
    "Relay": "A → AB → B (순차 전달)",
    "Sequential_Relay": "A → AB → B (순차 전달)",
    "Split": "AB → A → B (분리)",
    "Accumulation": "A → AB → ABC (누적 등장)",
    "Convergence": "A → B → AB (수렴)",
    "Sliding_Window": "AB → BC → C (슬라이딩)",
    "Reduction": "ABC → AB → A (점진 퇴장)",
    "Reverse_Relay": "B → AB → A (역순 전달)",
}

LIKERT_KR = {
    1: "1 — 전혀 안 맞음",
    2: "2 — 부족함",
    3: "3 — 보통",
    4: "4 — 양호",
    5: "5 — 완벽히 일치",
}

# ---------------------------------------------------------------------------
# Custom CSS — compact layout, mobile-friendly
# ---------------------------------------------------------------------------
CUSTOM_CSS = """
<style>
    /* Smaller image captions */
    .stImage > div > div > p { font-size: 0.75rem !important; margin: 0 !important; }
    /* Compact radio buttons */
    .stRadio > div { gap: 0.2rem !important; }
    .stRadio label { font-size: 0.85rem !important; }
    /* Tighter section spacing */
    .block-container { padding-top: 1.5rem !important; padding-bottom: 1rem !important; }
    /* Card style for video groups */
    .video-card {
        border: 2px solid #e0e0e0;
        border-radius: 10px;
        padding: 8px 10px 4px 10px;
        margin-bottom: 8px;
        background: #fafafa;
    }
    .video-card-a { border-color: #4285F4; }
    .video-card-b { border-color: #EA4335; }
    .video-label {
        font-weight: 700;
        font-size: 1rem;
        margin-bottom: 4px;
        text-align: center;
    }
    .label-a { color: #4285F4; }
    .label-b { color: #EA4335; }
    /* Presence matrix */
    .presence-table { width: 100%; border-collapse: collapse; font-size: 0.8rem; margin: 4px 0; }
    .presence-table th, .presence-table td {
        border: 1px solid #ccc; padding: 3px 6px; text-align: center;
    }
    .present { background: #C8E6C9; }
    .absent { background: #FFCDD2; }
    /* Info box */
    .scenario-info {
        background: #F3F4F6; border-radius: 8px; padding: 10px 14px; margin-bottom: 10px;
        font-size: 0.9rem; line-height: 1.5;
    }
    /* Mobile: stack columns */
    @media (max-width: 768px) {
        .stColumns > div { min-width: 100% !important; }
        .video-card { padding: 6px; }
    }
</style>
"""

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
# Helpers
# ---------------------------------------------------------------------------

def presence_html(s_matrix, entities, title="S*"):
    """Compact colored presence table."""
    n_shots = len(s_matrix)
    n_ents = len(s_matrix[0]) if n_shots else 0
    ent_labels = entities[:n_ents]

    hdr = "<tr><th></th>" + "".join(f"<th>{e}</th>" for e in ent_labels) + "</tr>"
    rows = ""
    for i in range(n_shots):
        cells = ""
        for j in range(n_ents):
            v = s_matrix[i][j] if j < len(s_matrix[i]) else 0
            cls = "present" if v == 1 else "absent"
            txt = "O" if v == 1 else "X"
            cells += f"<td class='{cls}'>{txt}</td>"
        rows += f"<tr><td><b>Shot {i+1}</b></td>{cells}</tr>"
    return f"<table class='presence-table'>{hdr}{rows}</table>"


def render_shots_inline(model, vid, col):
    """Render 3 shots in a single horizontal row inside col."""
    if model == "storydiff":
        kf_dir = KEYFRAMES_DIR / vid
        imgs = []
        for i in range(1, 4):
            p = kf_dir / f"shot0{i}.jpg"
            if not p.exists():
                p = kf_dir / f"shot0{i}.png"
            imgs.append(str(p) if p.exists() else None)

        subcols = col.columns(3)
        for i, (sc, img) in enumerate(zip(subcols, imgs)):
            if img:
                sc.image(img, caption=f"Shot {i+1}", use_container_width=True)
            else:
                sc.caption(f"Shot {i+1}: N/A")
    else:
        col.info(f"{MODEL_LABELS.get(model, model)} 키프레임 미제공 (클라우드 한정)")


# ---------------------------------------------------------------------------
# Page 1: 대시보드
# ---------------------------------------------------------------------------

def page_dashboard():
    st.markdown("## 📊 STEM 평가 결과 대시보드")
    data = load_eval_results()
    df = pd.DataFrame(data)

    # --- 요약 카드 ---
    cols = st.columns(4)
    for i, m in enumerate(MODELS):
        sub = df[df["method"] == m]
        ptm = sub["match_prescribed"].mean()
        epa = sub["epa"].mean()
        cols[i].metric(MODEL_LABELS[m], f"PTM {ptm:.3f}", f"EPA {epa:.3f}")

    # --- 바 차트 ---
    st.markdown("### 모델별 PTM & EPA")
    fig = go.Figure()
    for metric, color, name in [
        ("match_prescribed", "#4285F4", "PTM"),
        ("epa", "#34A853", "EPA"),
    ]:
        means = df.groupby("method")[metric].mean().reindex(MODELS)
        stds = df.groupby("method")[metric].std().reindex(MODELS)
        fig.add_trace(go.Bar(
            x=[MODEL_LABELS[m] for m in MODELS], y=means,
            error_y=dict(type="data", array=stds, visible=True),
            name=name, marker_color=color,
        ))
    fig.update_layout(barmode="group", yaxis_title="점수", height=350, margin=dict(t=30))
    st.plotly_chart(fig, use_container_width=True)

    # --- 패턴별 히트맵 ---
    st.markdown("### 패턴별 PTM 히트맵")
    pivot = df.pivot_table(values="match_prescribed", index="pattern", columns="method", aggfunc="mean")
    pivot = pivot[MODELS]
    pivot.columns = [MODEL_LABELS[m] for m in MODELS]
    fig2 = px.imshow(pivot.round(3), text_auto=True, color_continuous_scale="Blues",
                     labels=dict(color="PTM"), aspect="auto")
    fig2.update_layout(height=300, margin=dict(t=20))
    st.plotly_chart(fig2, use_container_width=True)

    # --- 실패 분포 ---
    st.markdown("### 실패 유형 분포 (영상 단위 %)")
    failure_rows = []
    for d in data:
        m = d["method"]
        E, dS = d["E_error"], d["delta_S_ideal"]
        types = set()
        for t in range(len(E)):
            for e in range(len(E[t])):
                if abs(E[t][e]) == 2:
                    types.add("IV 반전")
                elif dS[t][e] == -1 and E[t][e] > 0:
                    types.add("I 잔류")
                elif dS[t][e] == 1 and E[t][e] < 0:
                    types.add("II 미등장")
                elif dS[t][e] == 0 and E[t][e] != 0:
                    types.add("III 무단전환")
        if not types:
            types.add("없음 (완벽)")
        for ft in types:
            failure_rows.append({"모델": MODEL_LABELS[m], "실패 유형": ft})
    fdf = pd.DataFrame(failure_rows)
    fdf_pct = fdf.groupby(["모델", "실패 유형"]).size().reset_index(name="cnt")
    fdf_pct["비율"] = fdf_pct.apply(
        lambda r: 100 * r["cnt"] / len([d for d in data if MODEL_LABELS[d["method"]] == r["모델"]]), axis=1)
    fig3 = px.bar(fdf_pct, x="모델", y="비율", color="실패 유형", barmode="group",
                  labels={"비율": "영상 비율 (%)"}, height=350)
    fig3.update_layout(margin=dict(t=20))
    st.plotly_chart(fig3, use_container_width=True)

    # --- 코히어런스-컴플라이언스 트레이드오프 ---
    st.markdown("### 일관성-준수 트레이드오프")
    ptm_means = df.groupby("method")["match_prescribed"].mean().reindex(MODELS)
    ic_vals = {"storydiff": 0.81, "echoshot": 0.85, "vgot": 0.88, "vic": 0.94}
    fig4 = go.Figure()
    for m in MODELS:
        fig4.add_trace(go.Scatter(
            x=[ptm_means[m]], y=[ic_vals[m]], mode="markers+text",
            text=[MODEL_LABELS[m]], textposition="top center",
            marker=dict(size=14), name=MODEL_LABELS[m],
        ))
    fig4.update_layout(
        xaxis_title="PTM (내러티브 준수도) →", yaxis_title="Identity Consistency →",
        height=350, showlegend=False, margin=dict(t=20),
    )
    fig4.add_annotation(text="ρₛ = −1.0 (n=4)", x=0.15, y=0.92,
                        showarrow=False, font=dict(size=13, color="red"))
    st.plotly_chart(fig4, use_container_width=True)


# ---------------------------------------------------------------------------
# Page 2: 인간 평가
# ---------------------------------------------------------------------------

def page_human_eval():
    st.markdown("## 🎬 인간 평가: 내러티브 준수도")

    evaluator = st.sidebar.text_input("이름 입력", key="eval_name")
    if not evaluator:
        st.info("👈 왼쪽 사이드바에 이름을 입력해 주세요.")
        return

    prompts = load_prompts()
    _, by_prefix = build_eval_index()

    prompt_keys = list(prompts.keys())
    rng = random.Random(SEED)
    scenarios = rng.sample(prompt_keys, min(NUM_SCENARIOS, len(prompt_keys)))

    rng2 = random.Random(SEED)
    model_pairs = {}
    for sid in scenarios:
        ms = MODELS[:]
        rng2.shuffle(ms)
        pair = ms[:2]
        if rng2.random() < 0.5:
            pair = pair[::-1]
        model_pairs[sid] = {"A": pair[0], "B": pair[1]}

    if "eval_idx" not in st.session_state:
        st.session_state.eval_idx = 0
    idx = st.session_state.eval_idx

    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**진행률:** {idx} / {len(scenarios)}")
    st.sidebar.progress(idx / max(len(scenarios), 1))

    if idx >= len(scenarios):
        st.success("✅ 모든 평가를 완료했습니다! 감사합니다.")
        st.balloons()
        return

    sid = scenarios[idx]
    scenario = prompts[sid]
    meta = scenario["metadata"]
    pattern = meta["pattern_type"]
    pair = model_pairs[sid]
    prefix = "_".join(sid.split("_")[:2])
    entities = list(meta["core_entities"].values())

    # --- 시나리오 정보 박스 ---
    desc = PATTERN_DESC.get(pattern, pattern)
    info_html = f"""
    <div class='scenario-info'>
        <b>📌 시나리오 {idx+1}/{len(scenarios)}</b><br>
        <b>패턴:</b> {pattern.replace('Sequential_', '')} &nbsp;—&nbsp; {desc}<br>
        <b>등장 개체:</b> {', '.join(entities)}<br>
        <b>배경:</b> {meta.get('theme', 'N/A').replace('_', ' ')}
    </div>
    """
    st.markdown(info_html, unsafe_allow_html=True)

    # --- 처방된 존재 매트릭스 S* ---
    for m_name in [pair["A"], pair["B"]]:
        entry = by_prefix.get(m_name, {}).get(prefix)
        if entry and "S_ideal" in entry:
            st.markdown("**처방된 등장 매트릭스 (S*):**")
            st.markdown(presence_html(entry["S_ideal"], entry.get("entities", entities)), unsafe_allow_html=True)
            break

    # --- 영상 A / B 를 한 줄에 3 shot씩 ---
    st.markdown("---")

    for side, css_class, label_class, label_text in [
        ("A", "video-card-a", "label-a", "🔵 영상 A"),
        ("B", "video-card-b", "label-b", "🔴 영상 B"),
    ]:
        model = pair[side]
        entry = by_prefix.get(model, {}).get(prefix)
        vid = entry["vid"] if entry else ""

        st.markdown(f"<div class='video-card {css_class}'><div class='video-label {label_class}'>{label_text}</div>", unsafe_allow_html=True)

        if model == "storydiff" and entry:
            kf_dir = KEYFRAMES_DIR / vid
            imgs = []
            for i in range(1, 4):
                p = kf_dir / f"shot0{i}.jpg"
                if not p.exists():
                    p = kf_dir / f"shot0{i}.png"
                imgs.append(str(p) if p.exists() else None)

            c1, c2, c3 = st.columns(3)
            for ci, (col, img) in enumerate(zip([c1, c2, c3], imgs)):
                if img:
                    col.image(img, caption=f"Shot {ci+1}", use_container_width=True)
                else:
                    col.caption(f"Shot {ci+1}: N/A")

            # 관측된 S_obs 도 표시
            if entry and "S_obs" in entry:
                st.markdown(
                    f"<small><b>관측된 존재 (S_obs):</b></small>"
                    + presence_html(entry["S_obs"], entry.get("entities", entities), "S_obs"),
                    unsafe_allow_html=True,
                )
        elif entry:
            st.info(f"{MODEL_LABELS.get(model, model)} — 클라우드에서 키프레임 미제공")
            if "S_obs" in entry:
                st.markdown(
                    f"<small><b>관측된 존재 (S_obs):</b></small>"
                    + presence_html(entry["S_obs"], entry.get("entities", entities), "S_obs"),
                    unsafe_allow_html=True,
                )
            ptm_v = entry.get("match_prescribed", 0)
            epa_v = entry.get("epa", 0)
            st.caption(f"PTM: {ptm_v:.2f} | EPA: {epa_v:.2f}")

        st.markdown("</div>", unsafe_allow_html=True)

    # --- 평가 폼 ---
    st.markdown("---")
    st.markdown("### ✍️ 평가")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**🔵 영상 A 점수**")
        likert_a = st.radio("내러티브 준수도 (A):", [1, 2, 3, 4, 5],
                            format_func=lambda x: LIKERT_KR[x], key=f"la_{idx}", index=2,
                            label_visibility="collapsed")
    with c2:
        st.markdown("**🔴 영상 B 점수**")
        likert_b = st.radio("내러티브 준수도 (B):", [1, 2, 3, 4, 5],
                            format_func=lambda x: LIKERT_KR[x], key=f"lb_{idx}", index=2,
                            label_visibility="collapsed")

    st.markdown("**어느 영상이 내러티브를 더 잘 따르나요?**")
    pref = st.radio("선호:", ["A", "B", "비슷함"],
                    key=f"pref_{idx}", horizontal=True, label_visibility="collapsed")

    if st.button("다음 ➡️", type="primary", use_container_width=True):
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
# Page 3: 평가 결과
# ---------------------------------------------------------------------------

def page_eval_results():
    st.markdown("## 📋 인간 평가 결과")
    if not HUMAN_EVAL_OUTPUT.exists():
        st.info("아직 제출된 평가가 없습니다.")
        return
    with open(HUMAN_EVAL_OUTPUT) as f:
        results = json.load(f)
    if not results:
        st.info("아직 제출된 평가가 없습니다.")
        return

    st.metric("총 평가 수", len(results))
    rdf = pd.DataFrame(results)

    # 모델별 Likert 점수
    rows = []
    for _, r in rdf.iterrows():
        rows.append({"모델": MODEL_LABELS.get(r["model_a"], r["model_a"]), "점수": r["likert_a"]})
        rows.append({"모델": MODEL_LABELS.get(r["model_b"], r["model_b"]), "점수": r["likert_b"]})
    scores_df = pd.DataFrame(rows)
    st.markdown("### 모델별 평균 Likert 점수")
    summary = scores_df.groupby("모델")["점수"].agg(["mean", "std", "count"]).round(2)
    summary.columns = ["평균", "표준편차", "평가 수"]
    st.dataframe(summary, use_container_width=True)

    # 선호도
    st.markdown("### 모델별 승리 횟수")
    wins = defaultdict(int)
    for _, r in rdf.iterrows():
        if r["preference"] == "A":
            wins[r["model_a"]] += 1
        elif r["preference"] == "B":
            wins[r["model_b"]] += 1
    wins_df = pd.DataFrame([
        {"모델": MODEL_LABELS.get(m, m), "승리": wins.get(m, 0)} for m in MODELS
    ])
    st.dataframe(wins_df, use_container_width=True)

    # 평가자별 수
    st.markdown("### 평가자별 제출 수")
    evaluator_counts = rdf["evaluator"].value_counts().reset_index()
    evaluator_counts.columns = ["평가자", "평가 수"]
    st.dataframe(evaluator_counts, use_container_width=True)


# ---------------------------------------------------------------------------
# 안내 페이지
# ---------------------------------------------------------------------------

def page_guide():
    st.markdown("""
    ## 📖 평가 안내

    ### 평가 목적
    생성된 멀티-샷 영상이 **지정된 내러티브 패턴**을 얼마나 잘 따르는지 평가합니다.

    ### 평가 방법
    1. **시나리오 정보**를 확인하세요 (패턴, 등장 개체, 배경)
    2. **처방된 등장 매트릭스 S***를 확인하세요 (각 Shot에서 누가 있어야 하는지)
    3. **영상 A**와 **영상 B**의 3개 Shot 이미지를 비교하세요
    4. 각 영상에 1~5점 Likert 점수를 부여하세요
    5. 어느 영상이 더 나은지 선택하세요
    6. **다음** 버튼으로 진행하세요

    ### 점수 기준
    | 점수 | 의미 |
    |------|------|
    | 1 | 전혀 안 맞음 — 처방된 개체 등장/퇴장이 전혀 반영되지 않음 |
    | 2 | 부족함 — 일부만 맞고 대부분 틀림 |
    | 3 | 보통 — 절반 정도 맞음 |
    | 4 | 양호 — 대부분 맞지만 일부 오류 |
    | 5 | 완벽 — 모든 전환이 정확히 일치 |

    ### 패턴 설명
    | 패턴 | 구조 | 설명 |
    |------|------|------|
    | Relay | A → AB → B | A 먼저, B 합류 후 A 퇴장 |
    | Split | AB → A → B | 함께 있다가 B 퇴장, A 퇴장+B 재등장 |
    | Accumulation | A → AB → ABC | 개체가 하나씩 추가 |
    | Convergence | A → B → AB | A 퇴장, B 등장, 둘 다 등장 |
    | Sliding Window | AB → BC → C | 개체가 순차적으로 교체 |
    | Reduction | ABC → AB → A | 개체가 하나씩 퇴장 |
    | Reverse Relay | B → AB → A | Relay의 역순 |
    """)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    st.set_page_config(
        page_title="DIAGONAL 내러티브 평가",
        page_icon="🎬",
        layout="wide",
    )
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    st.sidebar.markdown("# 🎬 DIAGONAL")
    st.sidebar.markdown("**멀티-샷 영상 내러티브 평가**")
    st.sidebar.markdown("---")

    page = st.sidebar.radio(
        "메뉴",
        ["📊 대시보드", "🎬 인간 평가", "📋 평가 결과", "📖 평가 안내"],
    )

    if page == "📊 대시보드":
        page_dashboard()
    elif page == "🎬 인간 평가":
        page_human_eval()
    elif page == "📋 평가 결과":
        page_eval_results()
    else:
        page_guide()


if __name__ == "__main__":
    main()
