"""
DIAGONAL 인간 평가 시스템
========================
논문 Section 5.4 재현:
  - 15명 평가자, 40 시나리오 × 3 모델쌍 = 120 비교/인
  - 5점 Likert (내러티브 준수도) + 선호도 (A/B/Tie)
  - Fleiss' κ, Spearman ρ (PTM↔Human), 승률, 무승부율
"""
import streamlit as st
import json, random, datetime, itertools
import numpy as np, pandas as pd
import plotly.express as px, plotly.graph_objects as go
from pathlib import Path
from collections import defaultdict, Counter
from scipy import stats as sp_stats

# ── 경로 ───────────────────────────────────────────────────────────────
APP_DIR = Path(__file__).parent
DATA_DIR = APP_DIR / "data"
PROMPTS_PATH = DATA_DIR / "prompts_subset.json"
EVAL_PATH = DATA_DIR / "vlm_fullscale_merged.json"
KF_DIR = DATA_DIR / "keyframes"
OUTPUT_PATH = DATA_DIR / "human_eval_results.json"

# ── 상수 ───────────────────────────────────────────────────────────────
MODELS = ["storydiff", "echoshot", "vgot", "vic"]
MLABEL = {"storydiff": "StoryDiffusion", "echoshot": "EchoShot",
          "vgot": "VGoT", "vic": "VIC"}
SEED = 42
N_SCEN = 40
N_PAIRS = 3  # 40 × 3 = 120
ALL_PAIRS = list(itertools.combinations(MODELS, 2))

PAT_NAME = {
    "Relay": "Relay: A → AB → B",
    "Sequential_Relay": "Relay: A → AB → B",
    "Split": "Split: AB → A → B",
    "Accumulation": "Accumulation: A → AB → ABC",
    "Convergence": "Convergence: A → B → AB",
    "Sliding_Window": "Sliding Window: AB → BC → C",
    "Reduction": "Reduction: ABC → AB → A",
    "Reverse_Relay": "Reverse Relay: B → AB → A",
}

CSS = """<style>
@import url('https://cdn.jsdelivr.net/gh/orioncactus/pretendard/dist/web/static/pretendard.css');

/* ── 글로벌: 모바일 다크모드에서 흰 글씨 방지 ── */
*, body, html,
.main, .main *, .block-container, .block-container *,
[data-testid="stMarkdownContainer"], [data-testid="stMarkdownContainer"] *,
[data-testid="stText"], [data-testid="stCaptionContainer"],
p, span, li, ol, ul, label, small, h1, h2, h3, h4, h5, h6, div {
    font-family: 'Pretendard', -apple-system, 'Segoe UI', sans-serif !important;
    color: #1A1A1A !important;
}
/* Streamlit 위젯 라벨도 강제 검정 */
[data-testid="stWidgetLabel"] label,
[data-testid="stWidgetLabel"] p,
.stRadio label, .stRadio span,
[data-baseweb="radio"] label { color: #1A1A1A !important; }

.main .block-container { max-width: 960px; padding: 0.8rem 1rem; }
[data-testid="stImage"] img { border-radius: 8px; box-shadow: 0 1px 4px rgba(0,0,0,.12); }

/* 카드 */
.card { border-radius: 12px; padding: 14px 16px 8px; margin-bottom: 12px; }
.card-a { background: linear-gradient(135deg, #EBF5FB 0%, #D6EAF8 100%); border: 2px solid #3498DB; }
.card-b { background: linear-gradient(135deg, #FDEDEC 0%, #FADBD8 100%); border: 2px solid #E74C3C; }
.card-title { font-weight: 800; font-size: 1.1rem; margin-bottom: 8px; letter-spacing: -0.3px; }
.title-a { color: #2471A3 !important; }
.title-b { color: #C0392B !important; }

/* 시나리오 정보 박스 */
.info-box {
    background: linear-gradient(135deg, #F8F9FA 0%, #EBF5FB 100%);
    border-radius: 10px; padding: 14px 18px; margin-bottom: 14px;
    border-left: 5px solid #2980B9;
    line-height: 1.6;
}
.info-box b { color: #2C3E50 !important; }

/* S* 테이블 */
.ptable { border-collapse: collapse; font-size: 0.82rem; margin: 6px auto; }
.ptable th { background: #F0F3F5; color: #333 !important; }
.ptable th, .ptable td { border: 1px solid #CBD5E0; padding: 4px 10px; text-align: center; color: #333 !important; }
.p-yes { background: #A9DFBF; font-weight: 700; color: #1E8449 !important; }
.p-no  { background: #F5B7B1; color: #922B21 !important; }

/* 가이드 박스 */
.guide-box {
    background: #FFFBEA; border-radius: 10px; padding: 16px 20px;
    border: 1px solid #F0C36D; margin: 12px 0;
}
.guide-box h4 { margin: 0 0 8px 0; color: #92600F !important; }
.guide-box ol, .guide-box li { color: #333 !important; }
.guide-box ol { margin: 0; padding-left: 20px; }
.guide-box li { margin-bottom: 4px; line-height: 1.5; }
.guide-box p { color: #555 !important; }

/* 가이드라인 박스 */
.guideline-box {
    background: #FFF5F5; border-radius: 10px; padding: 16px 20px;
    border: 1px solid #E57373; margin: 12px 0;
}
.guideline-box h4 { margin: 0 0 10px 0; color: #C62828 !important; }
.guideline-box p, .guideline-box b, .guideline-box li { color: #333 !important; }
.guideline-box ul { margin: 0; padding-left: 18px; }
.guideline-box li { margin-bottom: 6px; line-height: 1.5; }

/* 스코어보드 */
.score-card {
    text-align: center; padding: 12px; border-radius: 10px;
    background: #F8F9FA; border: 1px solid #E0E0E0;
}
.score-val { font-size: 1.8rem; font-weight: 800; color: #2C3E50 !important; }
.score-lbl { font-size: 0.8rem; color: #7F8C8D !important; margin-top: 2px; }

/* 반응형 */
@media (max-width: 768px) {
    .main .block-container { padding: 0.4rem 0.6rem; }
    [data-testid="column"] { padding: 0 3px !important; }
    .card { padding: 10px 10px 6px; }
    .info-box { padding: 10px 12px; }
}
</style>"""


# ── 데이터 로드 ─────────────────────────────────────────────────────────
@st.cache_data
def load_eval():
    with open(EVAL_PATH) as f:
        return json.load(f)

@st.cache_data
def load_prompts():
    with open(PROMPTS_PATH) as f:
        return json.load(f)

@st.cache_data
def build_index():
    data = load_eval()
    bp = {}
    for d in data:
        pfx = "_".join(d["vid"].split("_")[:2])
        bp.setdefault(d["method"], {})[pfx] = d
    return bp

@st.cache_data
def build_trials():
    """40 시나리오 × 3 모델쌍 = 120 비교 (결정적 셔플)."""
    prompts = load_prompts()
    keys = list(prompts.keys())
    rng = random.Random(SEED)
    scenarios = rng.sample(keys, min(N_SCEN, len(keys)))

    trials = []
    rng2 = random.Random(SEED + 1)
    for sid in scenarios:
        pool = list(ALL_PAIRS)
        rng2.shuffle(pool)
        for a, b in pool[:N_PAIRS]:
            if rng2.random() < 0.5:
                a, b = b, a
            trials.append({"scenario": sid, "model_a": a, "model_b": b})

    rng3 = random.Random(SEED + 2)
    rng3.shuffle(trials)
    return trials


# ── 유틸리티 ────────────────────────────────────────────────────────────
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
        rows += f"<tr><td><b>Shot {i+1}</b></td>{cells}</tr>"
    return f"<table class='ptable'>{h}{rows}</table>"


def find_shots(model, vid):
    d = KF_DIR / f"{model}_{vid}"
    imgs = []
    for i in range(1, 4):
        found = False
        for ext in [".jpg", ".png"]:
            p = d / f"shot{i}{ext}"
            if p.exists():
                imgs.append(str(p)); found = True; break
        if not found:
            for ext in [".jpg", ".png"]:
                p = d / f"shot0{i}{ext}"
                if p.exists():
                    imgs.append(str(p)); found = True; break
        if not found:
            imgs.append(None)
    return imgs


def show_shots(model, vid, container):
    imgs = find_shots(model, vid)
    cols = container.columns(3, gap="small")
    for i, (c, img) in enumerate(zip(cols, imgs)):
        if img:
            c.image(img, use_container_width=True)
            c.caption(f"Shot {i+1}")
        else:
            c.markdown(
                f"<div style='background:#F0F0F0;border-radius:8px;height:110px;"
                f"display:flex;align-items:center;justify-content:center;"
                f"color:#999;font-size:0.85rem;'>Shot {i+1}<br>이미지 없음</div>",
                unsafe_allow_html=True,
            )


def load_results():
    if OUTPUT_PATH.exists():
        with open(OUTPUT_PATH) as f:
            data = json.load(f)
        return data if data else []
    return []


def save_results(results):
    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


def fleiss_kappa(mat):
    n_items, n_cats = mat.shape
    n_raters = int(mat.sum(axis=1)[0])
    if n_raters < 2:
        return float("nan")
    p_j = mat.sum(axis=0) / (n_items * n_raters)
    Pi = ((mat ** 2).sum(axis=1) - n_raters) / (n_raters * (n_raters - 1))
    P_bar = Pi.mean()
    Pe = (p_j ** 2).sum()
    if abs(1 - Pe) < 1e-10:
        return 1.0 if abs(P_bar - 1.0) < 1e-10 else 0.0
    return float((P_bar - Pe) / (1 - Pe))


# ═══════════════════════════════════════════════════════════════════════
#  페이지: 대시보드
# ═══════════════════════════════════════════════════════════════════════
def page_dashboard():
    st.markdown("## 📊 자동 평가 대시보드")
    data = load_eval()
    df = pd.DataFrame(data)

    # 요약 카드
    cols = st.columns(4)
    for i, m in enumerate(MODELS):
        s = df[df["method"] == m]
        cols[i].metric(MLABEL[m],
                       f"PTM {s['match_prescribed'].mean():.3f}",
                       f"EPA {s['epa'].mean():.3f}")

    # 바 차트
    fig = go.Figure()
    for metric, color, nm in [("match_prescribed", "#3498DB", "PTM"),
                               ("epa", "#2ECC71", "EPA")]:
        mu = df.groupby("method")[metric].mean().reindex(MODELS)
        sd = df.groupby("method")[metric].std().reindex(MODELS)
        fig.add_trace(go.Bar(
            x=[MLABEL[m] for m in MODELS], y=mu,
            error_y=dict(type="data", array=sd, visible=True),
            name=nm, marker_color=color,
        ))
    fig.update_layout(barmode="group", yaxis_title="점수", height=300,
                      margin=dict(t=10, b=30, l=40, r=10))
    st.plotly_chart(fig, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### 패턴별 PTM 히트맵")
        pv = df.pivot_table("match_prescribed", "pattern", "method", "mean")[MODELS]
        pv.columns = [MLABEL[m] for m in MODELS]
        fig2 = px.imshow(pv.round(3), text_auto=True, color_continuous_scale="Blues",
                         labels=dict(color="PTM"), aspect="auto")
        fig2.update_layout(height=260, margin=dict(t=10, b=10))
        st.plotly_chart(fig2, use_container_width=True)
    with c2:
        st.markdown("#### 일관성–준수 트레이드오프")
        ptm_mu = df.groupby("method")["match_prescribed"].mean().reindex(MODELS)
        ic = {"storydiff": 0.81, "echoshot": 0.85, "vgot": 0.88, "vic": 0.94}
        fig4 = go.Figure()
        for m in MODELS:
            fig4.add_trace(go.Scatter(
                x=[ptm_mu[m]], y=[ic[m]], mode="markers+text",
                text=[MLABEL[m]], textposition="top center",
                marker=dict(size=14), showlegend=False,
            ))
        fig4.update_layout(xaxis_title="PTM →", yaxis_title="IC →",
                           height=260, margin=dict(t=10, b=30))
        st.plotly_chart(fig4, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════
#  페이지: 인간 평가 (12개 배치, 스크롤 방식)
# ═══════════════════════════════════════════════════════════════════════
BATCH_SIZE = 12

def page_eval():
    prompts = load_prompts()
    bp = build_index()
    trials = build_trials()
    total = len(trials)

    # ── 사이드바 ──
    evaluator = st.sidebar.text_input("✏️ 평가자 이름", key="nm",
                                       placeholder="예: 홍길동")
    if not evaluator:
        st.markdown("## 🎬 인간 평가")
        st.markdown("""
<div class='guide-box'>
<h4>📋 평가 안내</h4>
<ol>
<li><b>사이드바에 이름 입력</b> 후 시작</li>
<li>각 비교에서 <b>처방된 등장 패턴(S*)</b>을 확인하세요</li>
<li><b>영상 A</b>와 <b>영상 B</b>의 Shot 3개를 비교합니다</li>
<li>각 영상이 패턴을 얼마나 따르는지 <b>5점 척도</b>로 평가</li>
<li><b>종합 판단</b> 4가지 중 하나를 선택</li>
<li>아래로 스크롤하며 12개씩 평가 → <b>[이 페이지 제출]</b></li>
</ol>
<p style="margin-top:10px;color:#666;">총 <b>120개</b> 비교 · 12개씩 10페이지</p>
</div>

<div class='guideline-box'>
<h4>🚨 평가 가이드라인: 무엇을 기준으로 점수를 매기나요?</h4>
<p>본 평가는 영상의 <b>화질이나 미적 아름다움이 아닌</b>, <b>'지시된 개체가 제때 나타나고 제때 사라졌는가(서사 준수도)'</b>를 평가합니다.</p>
<ul>
<li><b>5점 (완벽)</b> — 처방 패턴(S*)의 타이밍에 맞춰 개체의 등장과 퇴장이 완벽하게 이루어진 경우.</li>
<li><b>4점 (우수)</b> — 서사의 흐름(누가 들어오고 나가는지)은 맞췄으나, 퇴장해야 할 개체가 프레임 구석에 찰나의 잔상처럼 남는 등 아주 미세한 결함이 있는 경우.</li>
<li><b>3점 (보통)</b> — 샷이 3개이므로 전환(Transition)이 2번 일어납니다. 이 중 <b>한 번은 성공</b>했으나, <b>다른 한 번은 실패</b>한 경우. (예: A→AB는 맞췄으나, AB→B에서 실패)</li>
<li><b>2점 (미흡)</b> — 프롬프트에 있는 개체(예: 고양이)가 나오기는 하지만, 등장/퇴장 없이 처음부터 끝까지 가만히 있는 경우 (서사적 변화 실패).</li>
<li><b>1점 (실패)</b> — 지시된 개체가 아예 등장하지 않거나, 프롬프트를 완전히 무시한 엉뚱한 영상.</li>
</ul>
</div>""", unsafe_allow_html=True)
        return

    # 진행 상황
    all_results = load_results()
    my_done = {(r["vid"], r["model_a"], r["model_b"])
               for r in all_results if r.get("evaluator") == evaluator}
    completed = len(my_done)

    # 미완료 trial 목록
    remaining_trials = [
        (i, t) for i, t in enumerate(trials)
        if (t["scenario"], t["model_a"], t["model_b"]) not in my_done
    ]

    st.sidebar.markdown("---")
    st.sidebar.markdown(f"### 진행: **{completed}** / {total}")
    st.sidebar.progress(min(completed / max(total, 1), 1.0))
    st.sidebar.caption(f"남은 비교: {len(remaining_trials)}개")

    if not remaining_trials:
        st.markdown("## ✅ 평가 완료!")
        st.success("모든 120개 비교를 완료했습니다. 감사합니다! 🎉")
        st.balloons()
        return

    # 현재 배치 (12개)
    batch = remaining_trials[:BATCH_SIZE]
    batch_page = completed // BATCH_SIZE + 1
    total_pages = (total + BATCH_SIZE - 1) // BATCH_SIZE

    st.markdown(f"## 🎬 평가 — 페이지 {batch_page}/{total_pages}")
    st.caption(f"아래 {len(batch)}개 비교를 스크롤하며 평가한 후, 맨 아래 **[제출]** 버튼을 눌러주세요.")

    likert_labels = [
        "1 — 실패 (대본 완전 무시 / 엉뚱한 개체)",
        "2 — 미흡 (개체는 나오나 서사 변화 없음)",
        "3 — 보통 (전환 2번 중 1번만 성공)",
        "4 — 우수 (흐름은 맞으나 미세 잔상/오류)",
        "5 — 완벽 (등장/퇴장 타이밍 완벽 일치)",
    ]

    pref_options = ["🔵 A가 나음", "🔴 B가 나음",
                    "✅ 둘 다 잘함", "❌ 둘 다 실패"]

    # ── 배치 내 각 trial 렌더링 ──
    for batch_idx, (global_idx, trial) in enumerate(batch):
        sid = trial["scenario"]
        sc = prompts.get(sid, {})
        meta = sc.get("metadata", {})
        pat = meta.get("pattern_type", "Unknown")
        ma, mb = trial["model_a"], trial["model_b"]
        pfx = "_".join(sid.split("_")[:2])
        ents = list(meta.get("core_entities", {}).values())
        item_num = completed + batch_idx + 1

        st.markdown("---")

        # 시나리오 정보
        st.markdown(f"""
<div class='info-box'>
    <b>#{item_num}</b> &nbsp;|&nbsp;
    <b>{PAT_NAME.get(pat, pat)}</b> &nbsp;|&nbsp;
    👤 {', '.join(ents) if ents else '?'} &nbsp;|&nbsp;
    🏠 {meta.get('theme','').replace('_',' ')}
</div>""", unsafe_allow_html=True)

        # S* 매트릭스
        for mn in [ma, mb]:
            e = bp.get(mn, {}).get(pfx)
            if e and "S_ideal" in e:
                st.markdown(
                    "<div style='text-align:center;'>"
                    "<small><b>🎯 처방 패턴 (S*)</b> — ● 등장 · 미등장</small>"
                    "</div>" + presence_table(e["S_ideal"], e.get("entities", ents)),
                    unsafe_allow_html=True,
                )
                break

        # 영상 A / B 나란히
        col_a, col_b = st.columns(2, gap="medium")
        with col_a:
            st.markdown(
                "<div class='card card-a'><div class='card-title title-a'>🔵 A</div>",
                unsafe_allow_html=True,
            )
            ea = bp.get(ma, {}).get(pfx)
            if ea:
                show_shots(ma, ea["vid"], st)
            st.markdown("</div>", unsafe_allow_html=True)
        with col_b:
            st.markdown(
                "<div class='card card-b'><div class='card-title title-b'>🔴 B</div>",
                unsafe_allow_html=True,
            )
            eb = bp.get(mb, {}).get(pfx)
            if eb:
                show_shots(mb, eb["vid"], st)
            st.markdown("</div>", unsafe_allow_html=True)

        # Likert + 선호도 (한 줄에 압축)
        q1, q2, q3 = st.columns([1, 1, 1.2], gap="medium")
        with q1:
            st.markdown("**🔵 A 준수도**")
            st.radio("A", [1, 2, 3, 4, 5],
                     format_func=lambda x: likert_labels[x-1],
                     key=f"la_{global_idx}", index=1,
                     label_visibility="collapsed")
        with q2:
            st.markdown("**🔴 B 준수도**")
            st.radio("B", [1, 2, 3, 4, 5],
                     format_func=lambda x: likert_labels[x-1],
                     key=f"lb_{global_idx}", index=1,
                     label_visibility="collapsed")
        with q3:
            st.markdown("**❓ 종합 판단**")
            st.radio("선호", pref_options,
                     key=f"pf_{global_idx}", index=3,
                     label_visibility="collapsed")

    # ── 배치 제출 버튼 ──
    st.markdown("---")
    st.markdown("")
    if st.button(f"📮 이 페이지 {len(batch)}개 제출하고 다음으로",
                 type="primary", use_container_width=True):
        pref_map = {
            "🔵 A가 나음": "A",
            "🔴 B가 나음": "B",
            "✅ 둘 다 잘함": "BothGood",
            "❌ 둘 다 실패": "MutualFail",
        }
        new_records = []
        for global_idx, trial in batch:
            la = st.session_state.get(f"la_{global_idx}", 2)
            lb = st.session_state.get(f"lb_{global_idx}", 2)
            pf = st.session_state.get(f"pf_{global_idx}", pref_options[3])
            new_records.append({
                "evaluator": evaluator,
                "timestamp": datetime.datetime.now().isoformat(),
                "vid": trial["scenario"],
                "pattern": prompts.get(trial["scenario"], {})
                           .get("metadata", {}).get("pattern_type", ""),
                "model_a": trial["model_a"],
                "model_b": trial["model_b"],
                "likert_a": la,
                "likert_b": lb,
                "preference": pref_map.get(pf, "A"),
            })
        results = load_results()
        results.extend(new_records)
        save_results(results)
        st.rerun()


# ═══════════════════════════════════════════════════════════════════════
#  페이지: 결과 분석
# ═══════════════════════════════════════════════════════════════════════
def page_results():
    st.markdown("## 📋 결과 분석")
    results = load_results()
    if not results:
        st.info("아직 제출된 평가가 없습니다. '평가하기' 탭에서 시작하세요.")
        return

    rdf = pd.DataFrame(results)
    n_evaluators = rdf["evaluator"].nunique()

    # ── 요약 카드 ──
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(f"<div class='score-card'><div class='score-val'>{n_evaluators}</div>"
                f"<div class='score-lbl'>평가자</div></div>", unsafe_allow_html=True)
    c2.markdown(f"<div class='score-card'><div class='score-val'>{len(rdf)}</div>"
                f"<div class='score-lbl'>총 평가</div></div>", unsafe_allow_html=True)
    mf_count = (rdf["preference"] == "MutualFail").sum()
    mf_pct = 100 * mf_count / max(len(rdf), 1)
    c3.markdown(f"<div class='score-card'><div class='score-val'>{mf_pct:.1f}%</div>"
                f"<div class='score-lbl'>동반실패율</div></div>", unsafe_allow_html=True)
    target = n_evaluators * 120
    prog = min(100 * len(rdf) / max(target, 1), 100)
    c4.markdown(f"<div class='score-card'><div class='score-val'>{prog:.0f}%</div>"
                f"<div class='score-lbl'>전체 진행률</div></div>", unsafe_allow_html=True)

    st.markdown("---")

    # ── 모델별 승률 ──
    st.markdown("### 🏆 모델별 승률")
    wins = defaultdict(lambda: {"승": 0, "패": 0, "둘다잘함": 0, "둘다실패": 0})
    for _, r in rdf.iterrows():
        a, b = r["model_a"], r["model_b"]
        pref = r["preference"]
        if pref == "A":
            wins[a]["승"] += 1; wins[b]["패"] += 1
        elif pref == "B":
            wins[b]["승"] += 1; wins[a]["패"] += 1
        elif pref == "BothGood":
            wins[a]["둘다잘함"] += 1; wins[b]["둘다잘함"] += 1
        elif pref == "MutualFail":
            wins[a]["둘다실패"] += 1; wins[b]["둘다실패"] += 1
        else:  # legacy "Tie"
            wins[a]["둘다잘함"] += 1; wins[b]["둘다잘함"] += 1

    rows = []
    for m in MODELS:
        t = sum(wins[m].values())
        wr = 100 * wins[m]["승"] / max(t, 1)
        rows.append({"모델": MLABEL[m], "승": wins[m]["승"], "패": wins[m]["패"],
                      "둘다잘함": wins[m]["둘다잘함"], "둘다실패": wins[m]["둘다실패"],
                      "승률": f"{wr:.1f}%"})
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # 승률 바 차트
    wr_vals = [100 * wins[m]["승"] / max(sum(wins[m].values()), 1)
               for m in MODELS]
    fig_wr = go.Figure(go.Bar(
        x=[MLABEL[m] for m in MODELS], y=wr_vals,
        marker_color=["#3498DB", "#E67E22", "#2ECC71", "#9B59B6"],
        text=[f"{v:.1f}%" for v in wr_vals], textposition="auto",
    ))
    fig_wr.update_layout(yaxis_title="승률 (%)", height=260,
                         margin=dict(t=10, b=30))
    st.plotly_chart(fig_wr, use_container_width=True)

    # ── Likert 점수 ──
    if "likert_a" in rdf.columns:
        st.markdown("---")
        st.markdown("### 📊 내러티브 준수도 (Likert 평균)")
        likert = defaultdict(list)
        for _, r in rdf.iterrows():
            likert[r["model_a"]].append(r["likert_a"])
            likert[r["model_b"]].append(r["likert_b"])

        human_means = {}
        lrows = []
        for m in MODELS:
            mu = np.mean(likert[m]) if likert[m] else float("nan")
            sd = np.std(likert[m]) if likert[m] else float("nan")
            human_means[m] = mu
            lrows.append({"모델": MLABEL[m], "평균": f"{mu:.2f}",
                          "표준편차": f"{sd:.2f}", "평가 수": len(likert[m])})
        st.dataframe(pd.DataFrame(lrows), use_container_width=True, hide_index=True)

        # Likert 바 차트
        fig_l = go.Figure(go.Bar(
            x=[MLABEL[m] for m in MODELS],
            y=[np.mean(likert[m]) if likert[m] else 0 for m in MODELS],
            error_y=dict(type="data",
                         array=[np.std(likert[m]) if likert[m] else 0 for m in MODELS],
                         visible=True),
            marker_color=["#3498DB", "#E67E22", "#2ECC71", "#9B59B6"],
        ))
        fig_l.update_layout(yaxis_title="Likert (1–5)", yaxis_range=[0, 5.5],
                            height=260, margin=dict(t=10, b=30))
        st.plotly_chart(fig_l, use_container_width=True)

        # ── Spearman ρ ──
        st.markdown("---")
        st.markdown("### 📈 PTM ↔ 인간 평가 상관관계")
        try:
            edf = pd.DataFrame(load_eval())
            ptm_means = edf.groupby("method")["match_prescribed"].mean().reindex(MODELS)
            h_arr = np.array([human_means.get(m, np.nan) for m in MODELS])
            valid = ~np.isnan(h_arr)
            if valid.sum() >= 3:
                rho, pval = sp_stats.spearmanr(ptm_means.values[valid], h_arr[valid])
                rc1, rc2 = st.columns(2)
                rc1.metric("Spearman ρ", f"{rho:.3f}")
                rc2.metric("p-value", f"{pval:.4f}")
                corr_df = pd.DataFrame({
                    "모델": [MLABEL[m] for m in MODELS],
                    "PTM (자동)": ptm_means.values.round(3),
                    "Likert (인간)": np.round(h_arr, 2),
                })
                st.dataframe(corr_df, use_container_width=True, hide_index=True)
        except Exception as e:
            st.warning(f"상관관계 계산 실패: {e}")

    # ── Fleiss' κ ──
    st.markdown("---")
    st.markdown("### 🤝 평가자 간 일치도 (Fleiss' κ)")
    if n_evaluators >= 2 and "likert_a" in rdf.columns:
        try:
            item_ratings = defaultdict(list)
            for _, r in rdf.iterrows():
                item_ratings[(r["vid"], r["model_a"], r["model_b"], "A")].append(r["likert_a"])
                item_ratings[(r["vid"], r["model_a"], r["model_b"], "B")].append(r["likert_b"])

            rater_counts = Counter(len(v) for v in item_ratings.values())
            target_n = rater_counts.most_common(1)[0][0]
            filtered = {k: v for k, v in item_ratings.items() if len(v) == target_n}

            if target_n >= 2 and filtered:
                mat = np.zeros((len(filtered), 5), dtype=int)
                for i, ratings in enumerate(filtered.values()):
                    for r in ratings:
                        mat[i, r - 1] += 1
                kappa = fleiss_kappa(mat)

                kc1, kc2, kc3 = st.columns(3)
                kc1.metric("Fleiss' κ", f"{kappa:.3f}")
                kc2.metric("평가 항목 수", len(filtered))
                kc3.metric("항목당 평가자", target_n)

                if kappa > 0.8:
                    st.success("거의 완벽한 일치 (Almost Perfect)")
                elif kappa > 0.6:
                    st.info("상당한 일치 (Substantial)")
                elif kappa > 0.4:
                    st.warning("보통 일치 (Moderate)")
                else:
                    st.error("낮은 일치 (Fair/Poor)")
            else:
                st.warning(f"같은 항목을 평가한 평가자가 부족합니다 (최대 {target_n}명).")
        except Exception as e:
            st.error(f"Fleiss' κ 계산 오류: {e}")
    else:
        st.info("2명 이상의 평가자가 필요합니다.")

    # ── 평가자별 현황 ──
    st.markdown("---")
    st.markdown("### 👥 평가자별 진행 현황")
    ev_summary = []
    for name, grp in rdf.groupby("evaluator"):
        ev_summary.append({
            "평가자": name,
            "완료 수": len(grp),
            "진행률": f"{100 * len(grp) / 120:.0f}%",
            "마지막 평가": grp["timestamp"].max()[:16].replace("T", " "),
        })
    st.dataframe(pd.DataFrame(ev_summary), use_container_width=True, hide_index=True)

    # ── 다운로드 ──
    st.markdown("---")
    st.download_button(
        "📥 결과 JSON 다운로드",
        data=json.dumps(results, indent=2, ensure_ascii=False),
        file_name="human_eval_results.json",
        mime="application/json",
    )


# ═══════════════════════════════════════════════════════════════════════
#  메인
# ═══════════════════════════════════════════════════════════════════════
def main():
    st.set_page_config(page_title="DIAGONAL 인간 평가", page_icon="🎬",
                       layout="centered")
    st.markdown(CSS, unsafe_allow_html=True)

    st.sidebar.markdown("# 🎬 DIAGONAL")
    st.sidebar.markdown("멀티샷 영상 내러티브 평가")
    st.sidebar.markdown("---")

    page = st.sidebar.radio(
        "메뉴",
        ["🎬 인간 평가하기", "📊 자동 평가 대시보드", "📋 결과 & 분석"],
    )

    if "🎬" in page:
        page_eval()
    elif "📊" in page:
        page_dashboard()
    else:
        page_results()


if __name__ == "__main__":
    main()
