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

ENT_KO = {
    "apprentice": "견습생", "barista": "바리스타", "cat": "고양이",
    "chef": "요리사", "cook": "요리사", "crab": "게",
    "customer": "손님", "delivery cyclist": "배달원", "dishwasher": "설거지사",
    "doctor": "의사", "dog": "개", "employee": "직원",
    "frisbee player": "프리스비선수", "grandchild": "손주", "grandfather": "할아버지",
    "guest": "손님", "guide": "가이드", "hiker": "등산객",
    "intern": "인턴", "jogger": "조깅하는사람", "librarian": "사서",
    "manager": "매니저", "mentor": "멘토", "mother": "엄마",
    "nurse": "간호사", "orderly": "간병인", "owner": "주인",
    "pastry chef": "제과사", "patient": "환자", "pedestrian": "행인",
    "police officer": "경찰", "pupil": "학생", "server": "서버",
    "spectator": "관중", "street performer": "거리공연자", "student": "학생",
    "surgeon": "외과의사", "swimmer": "수영선수", "toddler": "아기",
    "tourist": "관광객", "tutor": "튜터", "visitor": "방문객", "waiter": "웨이터",
}

CSS = """<style>
@import url('https://cdn.jsdelivr.net/gh/orioncactus/pretendard/dist/web/static/pretendard.css');

/* ── 글로벌: 폰트 + 밝은 테마 강제 ── */
* { font-family: 'Pretendard', -apple-system, 'Segoe UI', sans-serif !important; }

/* 전체 앱을 밝은 배경 + 검정 글씨로 강제 (모바일 다크모드 무시) */
html, body, [data-testid="stAppViewContainer"],
[data-testid="stApp"], .main, .main *,
[data-testid="stSidebar"], [data-testid="stSidebar"] * {
    background-color: #FFFFFF !important;
    color: #1A1A1A !important;
}
/* 사이드바 약간 구분 */
[data-testid="stSidebar"] { background-color: #F7F8FA !important; }

/* 입력 필드, 라디오 등 위젯 */
[data-testid="stWidgetLabel"] label,
[data-testid="stWidgetLabel"] p,
[data-baseweb="radio"] label,
[data-baseweb="radio"] span,
.stRadio label, .stRadio span,
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] span,
[data-testid="stMarkdownContainer"] b,
[data-testid="stMarkdownContainer"] li,
[data-testid="stCaptionContainer"] { color: #1A1A1A !important; }

/* 버튼 텍스트는 흰색 유지 */
button[kind="primary"], button[kind="primary"] p,
button[kind="primary"] span,
.stButton button[kind="primary"] { color: #FFFFFF !important; }

.main .block-container { max-width: 960px; padding: 0.8rem 1rem; }
[data-testid="stImage"] img { border-radius: 8px; box-shadow: 0 1px 4px rgba(0,0,0,.12); }

/* 카드 */
.card { border-radius: 12px; padding: 14px 16px 8px; margin-bottom: 12px; }
.card-a { background: linear-gradient(135deg, #EBF5FB 0%, #D6EAF8 100%) !important; border: 2px solid #3498DB; }
.card-b { background: linear-gradient(135deg, #FDEDEC 0%, #FADBD8 100%) !important; border: 2px solid #E74C3C; }
.card, .card * { color: #1A1A1A !important; }
.card-title { font-weight: 800; font-size: 1.1rem; margin-bottom: 8px; letter-spacing: -0.3px; }
.title-a { color: #2471A3 !important; }
.title-b { color: #C0392B !important; }

/* 시나리오 정보 박스 */
.info-box {
    background: linear-gradient(135deg, #F8F9FA 0%, #EBF5FB 100%) !important;
    border-radius: 10px; padding: 14px 18px; margin-bottom: 14px;
    border-left: 5px solid #2980B9;
    line-height: 1.6;
}
.info-box, .info-box * { color: #2C3E50 !important; }

/* S* 테이블 */
.ptable { border-collapse: collapse; font-size: 0.82rem; margin: 6px auto; }
.ptable th { background: #F0F3F5 !important; color: #333 !important; }
.ptable th, .ptable td { border: 1px solid #CBD5E0; padding: 4px 10px; text-align: center; color: #333 !important; background-color: #FFF !important; }
.p-yes { background: #A9DFBF !important; font-weight: 700; color: #1E8449 !important; }
.p-no  { background: #F5B7B1 !important; color: #922B21 !important; }

/* 가이드 박스 */
.guide-box {
    background: #FFFBEA !important; border-radius: 10px; padding: 16px 20px;
    border: 1px solid #F0C36D; margin: 12px 0;
}
.guide-box, .guide-box * { color: #333 !important; }
.guide-box h4 { margin: 0 0 8px 0; color: #92600F !important; }
.guide-box ol { margin: 0; padding-left: 20px; }
.guide-box li { margin-bottom: 4px; line-height: 1.5; }

/* 가이드라인 박스 */
.guideline-box {
    background: #FFF5F5 !important; border-radius: 10px; padding: 16px 20px;
    border: 1px solid #E57373; margin: 12px 0;
}
.guideline-box, .guideline-box * { color: #333 !important; }
.guideline-box h4 { margin: 0 0 10px 0; color: #C62828 !important; }
.guideline-box ul { margin: 0; padding-left: 18px; }
.guideline-box li { margin-bottom: 6px; line-height: 1.5; }

/* 제출 확인 박스 */
.submit-confirm, .submit-confirm * { color: #333 !important; }

/* 스코어보드 */
.score-card {
    text-align: center; padding: 12px; border-radius: 10px;
    background: #F8F9FA !important; border: 1px solid #E0E0E0;
}
.score-card, .score-card * { color: #333 !important; }
.score-val { font-size: 1.8rem; font-weight: 800; color: #2C3E50 !important; }
.score-lbl { font-size: 0.8rem; color: #7F8C8D !important; margin-top: 2px; }

/* 반응형 — 모바일에서도 columns 가로 유지 */
@media (max-width: 768px) {
    .main .block-container { padding: 0.4rem 0.6rem; }
    .card { padding: 10px 10px 6px; }
    .info-box { padding: 10px 12px; }

    /* Streamlit columns를 모바일에서도 flex-row 강제 */
    [data-testid="stHorizontalBlock"] {
        flex-wrap: nowrap !important;
        gap: 4px !important;
    }
    [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] {
        min-width: 0 !important;
        flex: 1 1 0 !important;
        padding: 0 2px !important;
    }
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
    """Shot 3장을 st.columns로 배치 (CSS로 모바일 3열 강제)."""
    imgs = find_shots(model, vid)
    cols = container.columns(3, gap="small")
    for i, (c, img) in enumerate(zip(cols, imgs)):
        if img:
            c.image(img, use_container_width=True)
            c.caption(f"Shot {i+1}")
        else:
            c.markdown(
                f"<div style='background:#F0F0F0;border-radius:8px;height:80px;"
                f"display:flex;align-items:center;justify-content:center;"
                f"color:#999;font-size:0.8rem;'>Shot {i+1}<br>없음</div>",
                unsafe_allow_html=True,
            )


def make_entity_flow(s_ideal, entities, korean=False):
    """S_ideal 매트릭스에서 실제 개체 흐름 문자열 생성.
    예: [[1,1],[1,0],[0,1]] + [dog, owner] → 'dog&owner → dog → owner'
    korean=True → '개&주인 → 개 → 주인'
    """
    parts = []
    for row in s_ideal:
        present = []
        for j in range(min(len(row), len(entities))):
            if row[j]:
                name = entities[j]
                if korean:
                    name = ENT_KO.get(name, name)
                present.append(name)
        parts.append("&".join(present) if present else "∅")
    return " → ".join(parts)


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

    # ── 시작 전 화면: 이름 입력 + 평가 시작 ──
    if "eval_started" not in st.session_state:
        st.session_state.eval_started = False
    if "eval_name" not in st.session_state:
        st.session_state.eval_name = ""

    if not st.session_state.eval_started:
        st.markdown("## 🎬 인간 평가")
        st.markdown("")

        # 이름 입력
        st.markdown("### ✏️ 평가자 정보")
        name_input = st.text_input(
            "이름을 입력하세요",
            placeholder="예: 홍길동",
            key="name_input_field",
        )

        st.markdown("")

        # 평가 안내
        st.markdown("""
<div class='guide-box'>
<h4>📋 평가 안내</h4>
<ol>
<li>위에 <b>이름 입력</b> 후 아래 <b>[평가 시작]</b> 버튼 클릭</li>
<li>각 비교에서 <b>처방된 등장 패턴(S*)</b>을 확인하세요</li>
<li><b>영상 A</b>와 <b>영상 B</b>의 Shot 3개를 비교합니다</li>
<li>각 영상이 패턴을 얼마나 따르는지 <b>5점 척도</b>로 평가</li>
<li><b>종합 판단</b> 4가지 중 하나를 선택</li>
<li>아래로 스크롤하며 12개씩 평가 → <b>[제출]</b></li>
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

        st.markdown("")

        # 시작 버튼
        if st.button("🚀 평가 시작", type="primary", use_container_width=True):
            if not name_input.strip():
                st.error("⚠️ 이름을 먼저 입력해주세요!")
            else:
                st.session_state.eval_name = name_input.strip()
                st.session_state.eval_started = True
                st.rerun()
        return

    # ── 평가 진행 중 ──
    evaluator = st.session_state.eval_name

    # 사이드바: 평가자 정보
    st.sidebar.markdown(f"### 👤 {evaluator}")
    if st.sidebar.button("🔙 처음으로 돌아가기"):
        st.session_state.eval_started = False
        st.session_state.eval_name = ""
        st.rerun()

    # 페이지 네비게이션 (session_state 기반)
    total_pages = (total + BATCH_SIZE - 1) // BATCH_SIZE
    if "eval_page" not in st.session_state:
        st.session_state.eval_page = 0  # 0-indexed

    page_idx = st.session_state.eval_page

    # 진행 상황 계산
    all_results = load_results()
    my_done = {(r["vid"], r["model_a"], r["model_b"])
               for r in all_results if r.get("evaluator") == evaluator}
    # 기존 답변을 (vid, model_a, model_b) → record 로 조회용
    my_answers = {}
    for r in all_results:
        if r.get("evaluator") == evaluator:
            my_answers[(r["vid"], r["model_a"], r["model_b"])] = r
    completed = len(my_done)

    st.sidebar.markdown("---")
    st.sidebar.markdown(f"### 진행: **{completed}** / {total}")
    st.sidebar.progress(min(completed / max(total, 1), 1.0))
    st.sidebar.markdown(f"📄 페이지 **{page_idx + 1}** / {total_pages}")

    # 현재 페이지 배치 (고정 순서)
    start = page_idx * BATCH_SIZE
    end = min(start + BATCH_SIZE, total)
    batch = [(i, trials[i]) for i in range(start, end)]

    st.markdown(f"## 🎬 평가 — 페이지 {page_idx + 1}/{total_pages}")
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
    pref_reverse = {"A": 0, "B": 1, "BothGood": 2, "MutualFail": 3}

    # ── 배치 내 각 trial 렌더링 ──
    for batch_idx, (global_idx, trial) in enumerate(batch):
        sid = trial["scenario"]
        sc = prompts.get(sid, {})
        meta = sc.get("metadata", {})
        pat = meta.get("pattern_type", "Unknown")
        ma, mb = trial["model_a"], trial["model_b"]
        pfx = "_".join(sid.split("_")[:2])
        ents = list(meta.get("core_entities", {}).values())
        item_num = start + batch_idx + 1

        # 기존 답변 있으면 기본값으로 사용
        prev = my_answers.get((sid, ma, mb))
        default_la = (prev["likert_a"] - 1) if prev else 1  # index (0-based)
        default_lb = (prev["likert_b"] - 1) if prev else 1
        default_pf = pref_reverse.get(prev["preference"], 3) if prev else 3

        st.markdown("---")

        # S* 에서 실제 개체 흐름 추출
        entity_flow_en = ""
        entity_flow_ko = ""
        s_ideal_data = None
        s_ideal_ents = ents
        for mn in [ma, mb]:
            e = bp.get(mn, {}).get(pfx)
            if e and "S_ideal" in e:
                s_ideal_data = e["S_ideal"]
                s_ideal_ents = e.get("entities", ents)
                entity_flow_en = make_entity_flow(s_ideal_data, s_ideal_ents, korean=False)
                entity_flow_ko = make_entity_flow(s_ideal_data, s_ideal_ents, korean=True)
                break

        # 시나리오 정보 + 실제 개체 흐름 (EN + KO)
        flow_html = ""
        if entity_flow_en:
            flow_html = (
                f"<br>🎯 <b>정답:</b> "
                f"<span style='font-size:1.05em;'>{entity_flow_ko}</span>"
                f" <span style='font-size:0.85em;color:#666 !important;'>({entity_flow_en})</span>"
            )
        st.markdown(f"""
<div class='info-box'>
    <b>#{item_num}</b> &nbsp;|&nbsp;
    <b>{PAT_NAME.get(pat, pat)}</b> &nbsp;|&nbsp;
    👤 {', '.join(ents) if ents else '?'} &nbsp;|&nbsp;
    🏠 {meta.get('theme','').replace('_',' ')}
    {flow_html}
</div>""", unsafe_allow_html=True)

        # S* 매트릭스
        if s_ideal_data:
            st.markdown(
                "<div style='text-align:center;'>"
                "<small><b>🎯 처방 패턴 (S*)</b> — ● 등장 · 미등장</small>"
                "</div>" + presence_table(s_ideal_data, s_ideal_ents),
                unsafe_allow_html=True,
            )

        # 정답 흐름 HTML (A/B 카드 안에 삽입)
        answer_tag = ""
        if entity_flow_ko:
            answer_tag = (
                f"<div style='font-size:0.85rem;margin:4px 0 6px;padding:4px 8px;"
                f"background:rgba(255,255,255,0.7);border-radius:6px;"
                f"color:#1A1A1A !important;'>"
                f"🎯 <b style='color:#1A1A1A !important;'>정답:</b> "
                f"<span style='color:#1A1A1A !important;'>{entity_flow_ko}</span></div>"
            )

        # ── 영상 A (풀 너비) ──
        st.markdown(
            f"<div class='card card-a'>"
            f"<div class='card-title title-a'>🔵 A</div>"
            f"{answer_tag}",
            unsafe_allow_html=True,
        )
        ea = bp.get(ma, {}).get(pfx)
        if ea:
            show_shots(ma, ea["vid"], st)
        st.markdown("</div>", unsafe_allow_html=True)

        # ── 영상 B (풀 너비) ──
        st.markdown(
            f"<div class='card card-b'>"
            f"<div class='card-title title-b'>🔴 B</div>"
            f"{answer_tag}",
            unsafe_allow_html=True,
        )
        eb = bp.get(mb, {}).get(pfx)
        if eb:
            show_shots(mb, eb["vid"], st)
        st.markdown("</div>", unsafe_allow_html=True)

        # ── 준수도 + 종합판단 (세로 배치) ──
        edited_tag = " ✏️" if prev else ""
        st.markdown(f"**🔵 A 준수도**{edited_tag}")
        st.radio("A", [1, 2, 3, 4, 5],
                 format_func=lambda x: likert_labels[x-1],
                 key=f"la_{global_idx}", index=default_la,
                 horizontal=True, label_visibility="collapsed")

        st.markdown(f"**🔴 B 준수도**{edited_tag}")
        st.radio("B", [1, 2, 3, 4, 5],
                 format_func=lambda x: likert_labels[x-1],
                 key=f"lb_{global_idx}", index=default_lb,
                 horizontal=True, label_visibility="collapsed")

        st.markdown(f"**❓ 종합 판단**{edited_tag}")
        st.radio("선호", pref_options,
                 key=f"pf_{global_idx}", index=default_pf,
                 horizontal=True, label_visibility="collapsed")

    # ── 네비게이션 + 제출 ──
    st.markdown("---")

    # 이전 / 다음 페이지 버튼
    nav_left, nav_right = st.columns(2)
    with nav_left:
        if page_idx > 0:
            if st.button("← 이전 페이지", use_container_width=True):
                st.session_state.eval_page = page_idx - 1
                st.rerun()
    with nav_right:
        if page_idx < total_pages - 1:
            if st.button("다음 페이지 →", use_container_width=True):
                st.session_state.eval_page = page_idx + 1
                st.rerun()

    st.markdown("")

    # 제출 버튼
    if st.button(f"📮 이 페이지 {len(batch)}개 저장",
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
        # 기존 결과에서 이 평가자의 같은 trial 제거 후 새로 추가 (덮어쓰기)
        results = load_results()
        overwrite_keys = {(r["vid"], r["model_a"], r["model_b"]) for r in new_records}
        results = [r for r in results
                   if not (r.get("evaluator") == evaluator
                           and (r["vid"], r["model_a"], r["model_b"]) in overwrite_keys)]
        results.extend(new_records)
        save_results(results)
        # 자동으로 다음 페이지로 이동 (마지막이 아니면)
        if page_idx < total_pages - 1:
            st.session_state.eval_page = page_idx + 1
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
#  페이지: Failure 브라우저 (논문 figure 선별용)
# ═══════════════════════════════════════════════════════════════════════
def explain_failures_ko(fails, ents, s_ideal):
    """failure 코드를 S_ideal 기반으로 정확한 한글 설명으로 변환."""
    explanations = []
    for f in fails:
        parts = f.split("_")
        try:
            if f.startswith("missed_exit"):
                t_idx = int(parts[2][1:])  # t1 → 1 (transition index, 1-based)
                e_idx = int(parts[3][1:])  # e2 → 2
                ent_en = ents[e_idx] if e_idx < len(ents) else f"e{e_idx}"
                ent_ko = ENT_KO.get(ent_en, ent_en)
                # Shot t_idx에서는 있고, Shot t_idx+1에서는 없어야 하는데 계속 남아있음
                explanations.append(
                    f"Shot{t_idx}→{t_idx+1}: {ent_ko}({ent_en})가 "
                    f"Shot{t_idx+1}에서 사라져야 하는데 계속 남아있음 (잔류)"
                )
            elif f.startswith("missed_entry"):
                t_idx = int(parts[2][1:])
                e_idx = int(parts[3][1:])
                ent_en = ents[e_idx] if e_idx < len(ents) else f"e{e_idx}"
                ent_ko = ENT_KO.get(ent_en, ent_en)
                # Shot t_idx+1에서 등장해야 하는데 안 나옴
                explanations.append(
                    f"Shot{t_idx}→{t_idx+1}: {ent_ko}({ent_en})가 "
                    f"Shot{t_idx+1}에서 나와야 하는데 안 나옴 (망각)"
                )
            elif f.startswith("spurious"):
                t_idx = int(parts[1][1:])
                e_idx = int(parts[2][1:])
                ent_en = ents[e_idx] if e_idx < len(ents) else f"e{e_idx}"
                ent_ko = ENT_KO.get(ent_en, ent_en)
                # S_ideal 참조해서 정확한 설명
                if s_ideal and t_idx < len(s_ideal) and e_idx < len(s_ideal[0]):
                    before = s_ideal[t_idx][e_idx]      # Shot t_idx
                    after = s_ideal[t_idx + 1][e_idx] if t_idx + 1 < len(s_ideal) else before
                    if before == 1 and after == 1:
                        # 둘 다 있어야 하는데 변화가 생김 → Shot t에서 없었다가 나타남
                        explanations.append(
                            f"Shot{t_idx}→{t_idx+1}: {ent_ko}({ent_en})가 "
                            f"Shot{t_idx}에도 있어야 하는데 없다가 Shot{t_idx+1}에서 뒤늦게 나타남"
                        )
                    elif before == 0 and after == 0:
                        # 둘 다 없어야 하는데 나타남
                        explanations.append(
                            f"Shot{t_idx}→{t_idx+1}: {ent_ko}({ent_en})가 "
                            f"없어야 하는데 갑자기 나타남 (유령 등장)"
                        )
                    else:
                        explanations.append(
                            f"Shot{t_idx}→{t_idx+1}: {ent_ko}({ent_en})에 예상 밖 변화 발생"
                        )
                else:
                    explanations.append(
                        f"Shot{t_idx}→{t_idx+1}: {ent_ko}({ent_en})에 예상 밖 변화 발생"
                    )
            else:
                explanations.append(f)
        except (IndexError, ValueError):
            explanations.append(f)
    return explanations

SELECTED_PATH = DATA_DIR / "selected_failures.json"

def load_selected():
    if SELECTED_PATH.exists():
        with open(SELECTED_PATH) as f:
            return json.load(f)
    return []

def save_selected(sel):
    with open(SELECTED_PATH, "w") as f:
        json.dump(sel, f, indent=2, ensure_ascii=False)

def page_failure_browser():
    st.markdown("## 🔍 Failure 예시 브라우저")
    st.caption("논문 qualitative figure에 넣을 예시를 골라보세요. 썸네일 클릭 후 선택!")

    data = load_eval()

    # 선택된 항목
    if "fb_selected" not in st.session_state:
        st.session_state.fb_selected = load_selected()

    # failure 분류
    FAIL_TYPES = {
        "Type I: Context Bleeding (missed_exit)": "missed_exit",
        "Type II: Amnesia (missed_entry)": "missed_entry",
        "Type III: Spurious": "spurious",
        "Type IV: Reversal (E_error ±2)": "__reversal__",
    }
    FAIL_DESC = {
        "Type I: Context Bleeding (missed_exit)": (
            "🩸 <b>Type I — Context Bleeding (잔류 오류)</b><br>"
            "퇴장해야 할 개체가 다음 샷에서도 계속 남아있는 경우.<br>"
            "예: Relay(A→AB→B)에서 Shot 3에 A가 사라져야 하는데 여전히 보임."
        ),
        "Type II: Amnesia (missed_entry)": (
            "🧠 <b>Type II — Amnesia (망각 오류)</b><br>"
            "등장해야 할 개체가 해당 샷에 나타나지 않는 경우.<br>"
            "예: Relay(A→AB→B)에서 Shot 2에 B가 등장해야 하는데 안 보임."
        ),
        "Type III: Spurious": (
            "👻 <b>Type III — Spurious (유령 등장)</b><br>"
            "대본에 없는 개체가 갑자기 나타나는 경우.<br>"
            "예: Shot 1에 A만 있어야 하는데 B가 이미 보임."
        ),
        "Type IV: Reversal (E_error ±2)": (
            "🔄 <b>Type IV — Reversal (역전 오류)</b><br>"
            "등장/퇴장이 정반대로 일어난 경우.<br>"
            "예: A가 들어와야 하는데 오히려 나가고, B가 나가야 하는데 오히려 들어옴."
        ),
    }

    fc1, fc2, fc3 = st.columns(3)
    with fc1:
        sel_type = st.selectbox("Failure Type", list(FAIL_TYPES.keys()))

    # 선택된 type 설명
    st.markdown(
        f"<div style='background:#FFF8E1;border:1px solid #FFD54F;border-radius:8px;"
        f"padding:10px 14px;margin:8px 0;font-size:0.85rem;line-height:1.5;'>"
        f"{FAIL_DESC[sel_type]}</div>",
        unsafe_allow_html=True,
    )
    with fc2:
        sel_model = st.selectbox("모델", ["전체"] + MODELS)
    with fc3:
        sel_pattern = st.selectbox("패턴", ["전체", "Relay", "Split", "Accumulation",
                                            "Convergence", "Sliding_Window", "Reduction",
                                            "Reverse_Relay"])

    only_kf = st.checkbox("Keyframe 있는 것만", value=True)

    fail_key = FAIL_TYPES[sel_type]

    # 필터링
    candidates = []
    for r in data:
        if sel_model != "전체" and r["method"] != sel_model:
            continue
        if sel_pattern != "전체" and r.get("pattern") != sel_pattern:
            continue

        if fail_key == "__reversal__":
            has_rev = any(abs(v) == 2 for row in r.get("E_error", []) for v in row)
            if not has_rev:
                continue
            r["_sort"] = len(r.get("failures", []))
        else:
            matching = [f for f in r.get("failures", []) if fail_key in f]
            if not matching:
                continue
            other_fails = len([f for f in r.get("failures", []) if fail_key not in f])
            r["_sort"] = other_fails

        vid = r["vid"]
        model = r["method"]
        kf_dir = KF_DIR / f"{model}_{vid}"
        has_kf = kf_dir.exists() and any(kf_dir.iterdir()) if kf_dir.exists() else False
        r["_has_kf"] = has_kf

        if only_kf and not has_kf:
            continue

        candidates.append(r)

    candidates.sort(key=lambda x: (not x["_has_kf"], x["_sort"]))

    st.markdown(f"**{len(candidates)}개 후보** (keyframe 있는 것 우선, 깔끔한 예시 우선)")

    if not candidates:
        st.warning("조건에 맞는 결과가 없습니다.")
        return

    # 페이지네이션
    PAGE_SZ = 1000
    total_pg = (len(candidates) + PAGE_SZ - 1) // PAGE_SZ
    if "fb_page" not in st.session_state:
        st.session_state.fb_page = 0
    pg = st.session_state.fb_page
    page_cands = candidates[pg * PAGE_SZ : (pg + 1) * PAGE_SZ]

    if total_pg > 1:
        p1, p2, p3 = st.columns([1, 2, 1])
        with p1:
            if pg > 0 and st.button("← 이전 1000개"):
                st.session_state.fb_page -= 1
                st.rerun()
        p2.markdown(f"**페이지 {pg+1}/{total_pg}** ({pg*PAGE_SZ+1}~{min((pg+1)*PAGE_SZ, len(candidates))})")
        with p3:
            if pg < total_pg - 1 and st.button("다음 1000개 →"):
                st.session_state.fb_page += 1
                st.rerun()

    st.markdown("---")

    # 갤러리: 한 줄에 1개, 3 shot + 정보 + 선택 버튼
    sel_keys = {(s["vid"], s["method"], s.get("fail_type", "")) for s in st.session_state.fb_selected}

    for ci, r in enumerate(page_cands):
        vid = r["vid"]
        model = r["method"]
        ents = r.get("entities", [])
        fails = r.get("failures", [])
        item_key = (vid, model, sel_type.split(":")[0])
        is_selected = item_key in sel_keys

        border_color = "#27AE60" if is_selected else "#DDD"
        bg = "#E8F8F5" if is_selected else "#FAFAFA"

        # 정보 + 선택 버튼 (한 줄)
        n_other = len([f for f in fails if fail_key not in f]) if fail_key != "__reversal__" else 0
        clean_tag = "🟢" if n_other == 0 else f"🟡+{n_other}"

        # 정답 흐름 생성
        s_ideal = r.get("S_ideal")
        flow_en = make_entity_flow(s_ideal, ents, korean=False) if s_ideal else ""
        flow_ko = make_entity_flow(s_ideal, ents, korean=True) if s_ideal else ""

        info_col, btn_col = st.columns([4, 1])
        with info_col:
            st.markdown(
                f"<div style='font-size:0.8rem;line-height:1.4;padding:4px 8px;"
                f"background:{bg};border:1px solid {border_color};border-radius:6px;'>"
                f"<b>#{pg*PAGE_SZ+ci+1}</b> {clean_tag} "
                f"<b>{MLABEL.get(model, model)}</b> | "
                f"{r.get('pattern','')}<br>"
                f"🎯 정답: <b>{flow_ko}</b> ({flow_en})<br>"
                f"<span style='font-size:0.65rem;color:#888;'>{vid}</span>"
                f"</div>", unsafe_allow_html=True)
        with btn_col:
            btn_label = "✅ 선택됨" if is_selected else "☐ 선택"
            gidx = pg * PAGE_SZ + ci
            if st.button(btn_label, key=f"sel_{gidx}",
                         use_container_width=True):
                if is_selected:
                    st.session_state.fb_selected = [
                        s for s in st.session_state.fb_selected
                        if not (s["vid"] == vid and s["method"] == model
                                and s.get("fail_type", "") == sel_type.split(":")[0])
                    ]
                else:
                    st.session_state.fb_selected.append({
                        "vid": vid,
                        "method": model,
                        "pattern": r.get("pattern"),
                        "entities": ents,
                        "S_ideal": r.get("S_ideal"),
                        "S_obs": r.get("S_obs"),
                        "E_error": r.get("E_error"),
                        "failures": fails,
                        "fail_type": sel_type.split(":")[0],
                        "match_prescribed": r.get("match_prescribed"),
                        "epa": r.get("epa"),
                    })
                save_selected(st.session_state.fb_selected)
                st.rerun()

        # 3 Shot 표시
        show_shots(model, vid, st)
        st.markdown("")

    # 하단: 선택된 항목 요약
    st.markdown("---")
    st.markdown("### ⭐ 선택된 항목")
    selected = st.session_state.fb_selected
    if not selected:
        st.info("아직 선택된 항목이 없습니다.")
    else:
        for i, s in enumerate(selected):
            sc1, sc2 = st.columns([5, 1])
            sc1.markdown(
                f"**{s.get('fail_type','')}** | {MLABEL.get(s['method'], s['method'])} | "
                f"{s.get('pattern','')} | {', '.join(s.get('entities',[])[:3])} | "
                f"`{s['vid']}`"
            )
            if sc2.button("삭제", key=f"del_{i}"):
                st.session_state.fb_selected.pop(i)
                save_selected(st.session_state.fb_selected)
                st.rerun()

        st.download_button(
            "📥 선택 항목 JSON 다운로드",
            data=json.dumps(selected, indent=2, ensure_ascii=False),
            file_name="selected_failures.json",
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
        ["🎬 인간 평가하기", "📊 자동 평가 대시보드",
         "📋 결과 & 분석", "🔍 Failure 브라우저"],
    )

    if "🎬" in page:
        page_eval()
    elif "📊" in page:
        page_dashboard()
    elif "🔍" in page:
        page_failure_browser()
    else:
        page_results()


if __name__ == "__main__":
    main()
