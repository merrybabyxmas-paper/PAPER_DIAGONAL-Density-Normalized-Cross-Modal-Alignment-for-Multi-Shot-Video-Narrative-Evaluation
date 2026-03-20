"""
DIAGONAL 내러티브 평가
"""
import streamlit as st
import json, random, datetime, numpy as np, pandas as pd
import plotly.express as px, plotly.graph_objects as go
from pathlib import Path
from collections import defaultdict

APP_DIR = Path(__file__).parent
DATA_DIR = APP_DIR / "data"
PROMPTS_PATH = DATA_DIR / "prompts_subset.json"
EVAL_PATH = DATA_DIR / "vlm_fullscale_merged.json"
KF_DIR = DATA_DIR / "keyframes"
OUTPUT_PATH = DATA_DIR / "human_eval_results.json"

MODELS = ["storydiff", "echoshot", "vgot", "vic"]
MLABEL = {"storydiff": "StoryDiffusion", "echoshot": "EchoShot", "vgot": "VGoT", "vic": "VIC"}
SEED, N_SCEN = 42, 40

PAT_KR = {
    "Relay": "Relay: A → AB → B", "Sequential_Relay": "Relay: A → AB → B",
    "Split": "Split: AB → A → B", "Accumulation": "Accumulation: A → AB → ABC",
    "Convergence": "Convergence: A → B → AB", "Sliding_Window": "Sliding Window: AB → BC → C",
    "Reduction": "Reduction: ABC → AB → A", "Reverse_Relay": "Reverse Relay: B → AB → A",
}

CSS = """<style>
* { font-family: 'Pretendard', -apple-system, sans-serif !important; }
.main .block-container { max-width: 900px; padding: 1rem 1rem; }
[data-testid="stImage"] img { border-radius: 6px; }
.card { border-radius: 10px; padding: 10px 12px 6px; margin-bottom: 10px; }
.card-a { background: #EBF5FB; border: 2px solid #3498DB; }
.card-b { background: #FDEDEC; border: 2px solid #E74C3C; }
.card-title { font-weight: 800; font-size: 1.05rem; margin-bottom: 6px; }
.title-a { color: #2980B9; }
.title-b { color: #C0392B; }
.info-box { background: #F8F9FA; border-radius: 8px; padding: 12px 16px; margin-bottom: 12px;
  border-left: 4px solid #3498DB; }
.ptable { border-collapse: collapse; font-size: 0.78rem; margin: 4px 0; width: auto; }
.ptable th, .ptable td { border: 1px solid #bbb; padding: 2px 8px; text-align: center; }
.p-yes { background: #A9DFBF; font-weight: 700; }
.p-no { background: #F5B7B1; }
.conf-row { display: flex; gap: 6px; justify-content: center; margin: 8px 0; }
.conf-btn { padding: 6px 14px; border-radius: 20px; cursor: pointer;
  border: 2px solid #bbb; background: white; font-size: 0.85rem; }
@media (max-width: 640px) {
  .main .block-container { padding: 0.5rem; }
  [data-testid="column"] { padding: 0 2px !important; }
}
</style>"""

# ---- Data ----
@st.cache_data
def load_eval():
    with open(EVAL_PATH) as f: return json.load(f)

@st.cache_data
def load_prompts():
    with open(PROMPTS_PATH) as f: return json.load(f)

@st.cache_data
def build_index():
    data = load_eval()
    bp = {}
    for d in data:
        pfx = "_".join(d["vid"].split("_")[:2])
        bp.setdefault(d["method"], {})[pfx] = d
    return bp

# ---- Helpers ----
def presence_table(mat, ents):
    n = len(mat[0]) if mat else 0
    el = ents[:n]
    h = "<tr><th></th>" + "".join(f"<th>{e}</th>" for e in el) + "</tr>"
    rows = ""
    for i, row in enumerate(mat):
        cells = "".join(
            f"<td class='{'p-yes' if (row[j] if j < len(row) else 0) else 'p-no'}'>"
            f"{'●' if (row[j] if j < len(row) else 0) else '·'}</td>"
            for j in range(n)
        )
        rows += f"<tr><td><b>S{i+1}</b></td>{cells}</tr>"
    return f"<table class='ptable'>{h}{rows}</table>"

def find_shots(model, vid):
    d = KF_DIR / f"{model}_{vid}"
    imgs = []
    for i in range(1, 4):
        for ext in [".jpg", ".png"]:
            p = d / f"shot{i}{ext}"
            if p.exists():
                imgs.append(str(p)); break
        else:
            # Try old naming
            p2 = d / f"shot0{i}{ext}"
            for ext2 in [".jpg", ".png"]:
                p3 = d / f"shot0{i}{ext2}"
                if p3.exists():
                    imgs.append(str(p3)); break
            else:
                imgs.append(None)
    return imgs

def show_shots_row(model, vid, container):
    imgs = find_shots(model, vid)
    cols = container.columns(3, gap="small")
    for i, (c, img) in enumerate(zip(cols, imgs)):
        if img:
            c.image(img, use_container_width=True)
            c.caption(f"Shot {i+1}")
        else:
            c.markdown(f"<div style='background:#eee;border-radius:6px;height:100px;"
                       f"display:flex;align-items:center;justify-content:center;color:#999;'>"
                       f"Shot {i+1}<br>N/A</div>", unsafe_allow_html=True)


# ===========================================================================
# 대시보드
# ===========================================================================
def page_dashboard():
    st.markdown("## 📊 평가 결과 대시보드")
    data = load_eval()
    df = pd.DataFrame(data)

    # 요약 카드
    cols = st.columns(4)
    for i, m in enumerate(MODELS):
        s = df[df["method"] == m]
        cols[i].metric(MLABEL[m], f"{s['match_prescribed'].mean():.3f}",
                       f"EPA {s['epa'].mean():.3f}")

    # 바 차트
    fig = go.Figure()
    for metric, color, nm in [("match_prescribed","#3498DB","PTM"),("epa","#2ECC71","EPA")]:
        mu = df.groupby("method")[metric].mean().reindex(MODELS)
        sd = df.groupby("method")[metric].std().reindex(MODELS)
        fig.add_trace(go.Bar(x=[MLABEL[m] for m in MODELS], y=mu,
            error_y=dict(type="data",array=sd,visible=True), name=nm, marker_color=color))
    fig.update_layout(barmode="group", yaxis_title="점수", height=320, margin=dict(t=20,b=30))
    st.plotly_chart(fig, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### 패턴별 PTM")
        pv = df.pivot_table("match_prescribed", "pattern", "method", "mean")[MODELS]
        pv.columns = [MLABEL[m] for m in MODELS]
        fig2 = px.imshow(pv.round(3), text_auto=True, color_continuous_scale="Blues",
                         labels=dict(color="PTM"), aspect="auto")
        fig2.update_layout(height=280, margin=dict(t=10,b=10))
        st.plotly_chart(fig2, use_container_width=True)

    with c2:
        st.markdown("### 일관성-준수 트레이드오프")
        ptm_mu = df.groupby("method")["match_prescribed"].mean().reindex(MODELS)
        ic = {"storydiff":0.81,"echoshot":0.85,"vgot":0.88,"vic":0.94}
        fig4 = go.Figure()
        for m in MODELS:
            fig4.add_trace(go.Scatter(x=[ptm_mu[m]], y=[ic[m]], mode="markers+text",
                text=[MLABEL[m]], textposition="top center", marker=dict(size=13), showlegend=False))
        fig4.update_layout(xaxis_title="PTM →", yaxis_title="IC →",
                           height=280, margin=dict(t=10,b=30))
        fig4.add_annotation(text="ρₛ=−1.0", x=0.15, y=0.92, showarrow=False,
                            font=dict(size=12, color="red"))
        st.plotly_chart(fig4, use_container_width=True)


# ===========================================================================
# 인간 평가
# ===========================================================================
def page_eval():
    prompts = load_prompts()
    bp = build_index()
    keys = list(prompts.keys())
    rng = random.Random(SEED)
    scenarios = rng.sample(keys, min(N_SCEN, len(keys)))

    rng2 = random.Random(SEED)
    pairs = {}
    for sid in scenarios:
        ms = MODELS[:]
        rng2.shuffle(ms)
        p = ms[:2]
        if rng2.random() < 0.5: p = p[::-1]
        pairs[sid] = {"A": p[0], "B": p[1]}

    # Sidebar
    evaluator = st.sidebar.text_input("✏️ 이름", key="nm")
    if not evaluator:
        st.markdown("## 🎬 인간 평가")
        st.info("👈 사이드바에 이름을 입력하고 시작하세요.")
        st.markdown("""
        ### 평가 방법
        1. **시나리오 정보** 확인 (패턴, 개체, 배경)
        2. **영상 A / B** 의 3개 Shot 비교
        3. **어느 쪽이 내러티브를 더 잘 따르는지** 선택
        4. **확신도** 선택
        5. **다음** 클릭
        """)
        return

    if "idx" not in st.session_state: st.session_state.idx = 0
    idx = st.session_state.idx
    total = len(scenarios)

    st.sidebar.markdown(f"**{idx}/{total}** 완료")
    st.sidebar.progress(idx / max(total, 1))

    if idx >= total:
        st.markdown("## ✅ 평가 완료!")
        st.success("모든 시나리오를 평가했습니다. 감사합니다!")
        st.balloons()
        return

    sid = scenarios[idx]
    sc = prompts[sid]
    meta = sc["metadata"]
    pat = meta["pattern_type"]
    pair = pairs[sid]
    pfx = "_".join(sid.split("_")[:2])
    ents = list(meta["core_entities"].values())

    # 시나리오 정보
    st.markdown(f"""
    <div class='info-box'>
        <b>시나리오 {idx+1}/{total}</b> &nbsp;|&nbsp;
        <b>{pat.replace('Sequential_','')}</b>: {PAT_KR.get(pat, pat)}<br>
        👤 <b>개체:</b> {', '.join(ents)} &nbsp;|&nbsp;
        🏠 <b>배경:</b> {meta.get('theme','').replace('_',' ')}
    </div>""", unsafe_allow_html=True)

    # S* 매트릭스
    for mn in [pair["A"], pair["B"]]:
        e = bp.get(mn, {}).get(pfx)
        if e and "S_ideal" in e:
            st.markdown("**처방된 등장 (S*)** — " + presence_table(e["S_ideal"], e.get("entities", ents)),
                        unsafe_allow_html=True)
            break

    # 영상 A
    st.markdown("<div class='card card-a'><div class='card-title title-a'>🔵 영상 A</div>",
                unsafe_allow_html=True)
    ea = bp.get(pair["A"], {}).get(pfx)
    if ea:
        show_shots_row(pair["A"], ea["vid"], st)
    st.markdown("</div>", unsafe_allow_html=True)

    # 영상 B
    st.markdown("<div class='card card-b'><div class='card-title title-b'>🔴 영상 B</div>",
                unsafe_allow_html=True)
    eb = bp.get(pair["B"], {}).get(pfx)
    if eb:
        show_shots_row(pair["B"], eb["vid"], st)
    st.markdown("</div>", unsafe_allow_html=True)

    # 평가 질문 (2개만)
    st.markdown("---")

    pref = st.radio(
        "❓ **어느 영상이 내러티브 패턴을 더 잘 따르나요?**",
        ["🔵 A가 더 나음", "🔴 B가 더 나음", "⚖️ 비슷함"],
        key=f"pref_{idx}", horizontal=True,
    )

    conf = st.select_slider(
        "📏 **확신도**",
        options=["매우 불확실", "약간 불확실", "보통", "꽤 확실", "매우 확실"],
        value="보통", key=f"conf_{idx}",
    )

    if st.button("다음 ➡️", type="primary", use_container_width=True):
        pref_map = {"🔵 A가 더 나음": "A", "🔴 B가 더 나음": "B", "⚖️ 비슷함": "Tie"}
        conf_map = {"매우 불확실": 1, "약간 불확실": 2, "보통": 3, "꽤 확실": 4, "매우 확실": 5}
        rec = {
            "evaluator": evaluator,
            "ts": datetime.datetime.now().isoformat(),
            "vid": sid, "pattern": pat,
            "model_a": pair["A"], "model_b": pair["B"],
            "preference": pref_map[pref],
            "confidence": conf_map[conf],
        }
        results = []
        if OUTPUT_PATH.exists():
            with open(OUTPUT_PATH) as f: results = json.load(f)
        results.append(rec)
        with open(OUTPUT_PATH, "w") as f: json.dump(results, f, indent=2)
        st.session_state.idx = idx + 1
        st.rerun()


# ===========================================================================
# 결과
# ===========================================================================
def page_results():
    st.markdown("## 📋 평가 결과")
    if not OUTPUT_PATH.exists():
        st.info("아직 제출된 평가가 없습니다.")
        return
    with open(OUTPUT_PATH) as f: results = json.load(f)
    if not results:
        st.info("아직 제출된 평가가 없습니다.")
        return

    rdf = pd.DataFrame(results)
    st.metric("총 평가 수", len(results))

    # 모델별 승률
    wins = defaultdict(lambda: {"승": 0, "패": 0, "무": 0})
    for _, r in rdf.iterrows():
        ma, mb = r["model_a"], r["model_b"]
        if r["preference"] == "A":
            wins[ma]["승"] += 1; wins[mb]["패"] += 1
        elif r["preference"] == "B":
            wins[mb]["승"] += 1; wins[ma]["패"] += 1
        else:
            wins[ma]["무"] += 1; wins[mb]["무"] += 1

    wdf = pd.DataFrame([
        {"모델": MLABEL.get(m, m), **wins[m],
         "승률": f"{100*wins[m]['승']/max(wins[m]['승']+wins[m]['패']+wins[m]['무'],1):.0f}%"}
        for m in MODELS
    ])
    st.dataframe(wdf, use_container_width=True, hide_index=True)

    # 확신도 분포
    if "confidence" in rdf.columns:
        st.markdown("### 확신도 분포")
        fig = px.histogram(rdf, x="confidence", nbins=5, labels={"confidence": "확신도"})
        fig.update_layout(height=250, margin=dict(t=10))
        st.plotly_chart(fig, use_container_width=True)

    # 평가자별
    st.markdown("### 평가자별 제출")
    st.dataframe(rdf["evaluator"].value_counts().reset_index().rename(
        columns={"evaluator": "평가자", "count": "평가 수"}),
        use_container_width=True, hide_index=True)


# ===========================================================================
# Main
# ===========================================================================
def main():
    st.set_page_config(page_title="DIAGONAL 평가", page_icon="🎬", layout="centered")
    st.markdown(CSS, unsafe_allow_html=True)

    st.sidebar.markdown("# 🎬 DIAGONAL")
    st.sidebar.markdown("멀티샷 영상 내러티브 평가")
    st.sidebar.markdown("---")

    page = st.sidebar.radio("메뉴", ["📊 대시보드", "🎬 평가하기", "📋 결과 보기"])

    if "📊" in page: page_dashboard()
    elif "🎬" in page: page_eval()
    else: page_results()

if __name__ == "__main__":
    main()
