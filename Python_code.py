"""
ExamForge AI - AI 기반 시험 문제 생성 & 학습 플랫폼
단일 파일 버전 (Streamlit Cloud 배포용)
"""
import os
import io
import json
import random
import hashlib
import sqlite3
from datetime import datetime
from contextlib import contextmanager

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import altair as alt
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

# =========================================================================
# 0. 페이지 설정 & 전역 스타일
# =========================================================================
st.set_page_config(
    page_title="ExamForge AI",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.card { background: linear-gradient(135deg,#1A1F2E 0%,#232842 100%);
        padding:22px 24px; border-radius:16px;
        border:1px solid rgba(255,255,255,0.06);
        box-shadow:0 4px 16px rgba(0,0,0,0.25); margin-bottom:14px;}
.badge { display:inline-block; padding:4px 10px; border-radius:999px;
         font-size:12px; font-weight:600; margin-right:6px;}
.badge-purple { background:#7C5CFF22; color:#B9A6FF;}
.badge-green  { background:#22c55e22; color:#86efac;}
.badge-red    { background:#ef444422; color:#fca5a5;}
.stat-num { font-size:32px; font-weight:800; color:#FAFAFA;}
.stat-label{ font-size:13px; color:#9CA3AF;}
.q-title { font-size:18px; font-weight:700; line-height:1.5;}
</style>
""", unsafe_allow_html=True)


# =========================================================================
# 1. 환경설정
# =========================================================================
def get_api_key():
    """Streamlit Secrets → 환경변수 순으로 키 검색."""
    try:
        return st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY", ""))
    except Exception:
        return os.getenv("OPENAI_API_KEY", "")


def get_model():
    try:
        return st.secrets.get("OPENAI_MODEL", os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
    except Exception:
        return os.getenv("OPENAI_MODEL", "gpt-4o-mini")


OPENAI_API_KEY = get_api_key()
OPENAI_MODEL = get_model()
DB_PATH = "examforge.db"
MAX_PDF_CHARS = 60_000
QUESTION_COUNTS = [5, 10, 20, 30]
DIFFICULTIES = ["쉬움", "보통", "어려움"]
QUESTION_TYPES = ["개념 이해", "정의 암기", "응용 문제", "비교 분석", "종합 문제"]


# =========================================================================
# 2. 데이터베이스 (SQLite)
# =========================================================================
@contextmanager
def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    with get_conn() as conn:
        conn.executescript("""
        CREATE TABLE IF NOT EXISTS sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            pdf_name TEXT, difficulty TEXT, qtype TEXT,
            num_questions INTEGER, score INTEGER, total INTEGER,
            created_at TEXT
        );
        CREATE TABLE IF NOT EXISTS attempts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id INTEGER, question_json TEXT,
            user_answer TEXT, is_correct INTEGER
        );
        """)


def save_session(pdf_name, difficulty, qtype, attempts, score, total):
    with get_conn() as conn:
        cur = conn.execute(
            """INSERT INTO sessions(pdf_name,difficulty,qtype,num_questions,
                                    score,total,created_at)
               VALUES (?,?,?,?,?,?,?)""",
            (pdf_name, difficulty, qtype, total, score, total,
             datetime.now().isoformat()))
        sid = cur.lastrowid
        for a in attempts:
            conn.execute(
                """INSERT INTO attempts(session_id,question_json,
                                        user_answer,is_correct)
                   VALUES (?,?,?,?)""",
                (sid, json.dumps(a["question"], ensure_ascii=False),
                 a["user_answer"], int(a["is_correct"])))


def fetch_history(limit=100):
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM sessions ORDER BY created_at DESC LIMIT ?",
            (limit,)).fetchall()
        return [dict(r) for r in rows]


# =========================================================================
# 3. PDF 처리
# =========================================================================
@st.cache_data(show_spinner=False, ttl=3600)
def extract_pdf_text(file_bytes: bytes) -> str:
    """PyMuPDF → pdfplumber → pypdf 폴백."""
    # 1차: PyMuPDF
    try:
        import fitz
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        text = "\n".join(page.get_text("text") for page in doc)
        if text.strip():
            return preprocess(text)
    except Exception:
        pass
    # 2차: pdfplumber
    try:
        import pdfplumber
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            text = "\n".join((p.extract_text() or "") for p in pdf.pages)
        if text.strip():
            return preprocess(text)
    except Exception:
        pass
    # 3차: pypdf
    try:
        from pypdf import PdfReader
        reader = PdfReader(io.BytesIO(file_bytes))
        text = "\n".join((p.extract_text() or "") for p in reader.pages)
        return preprocess(text)
    except Exception as e:
        raise RuntimeError(f"PDF 추출 실패: {e}")


def preprocess(text: str) -> str:
    import re
    text = re.sub(r"\r", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"-\n(\w)", r"\1", text)
    return text.strip()


def truncate_for_llm(text: str, max_chars: int = MAX_PDF_CHARS) -> str:
    if len(text) <= max_chars:
        return text
    head = text[: int(max_chars * 0.7)]
    tail = text[-int(max_chars * 0.3):]
    return head + "\n...[중략]...\n" + tail


# =========================================================================
# 4. OpenAI 호출
# =========================================================================
def get_client():
    if not OPENAI_API_KEY:
        raise RuntimeError(
            "⚠️ OPENAI_API_KEY가 설정되지 않았습니다.\n"
            "Streamlit Cloud의 Settings → Secrets에 키를 등록하세요."
        )
    return OpenAI(api_key=OPENAI_API_KEY)


@retry(stop=stop_after_attempt(3),
       wait=wait_exponential(multiplier=1, min=2, max=10))
def chat_json(system: str, user: str, temperature: float = 0.4) -> dict:
    resp = get_client().chat.completions.create(
        model=OPENAI_MODEL,
        temperature=temperature,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    return json.loads(resp.choices[0].message.content)


@st.cache_data(show_spinner=False, ttl=3600)
def summarize_material(text: str) -> dict:
    text = truncate_for_llm(text)
    system = ("당신은 한국어 교육 전문가다. 강의자료를 학습자가 빠르게 "
              "파악하도록 정리하라. JSON으로만 응답하라.")
    user = f"""다음 강의자료를 JSON 스키마에 맞게 정리하라.

[강의자료]
{text}

[JSON 스키마]
{{
  "summary": "5∼8문장 요약",
  "keywords": ["핵심 키워드 10∼15개"],
  "top_concepts": [{{"concept":"개념명","description":"1∼2문장 설명"}}]
}}

규칙:
- top_concepts 정확히 10개
- 강의자료 외 내용 금지
"""
    return chat_json(system, user, 0.3)


def generate_questions(text, num, difficulty, qtype, prev_score=None):
    text = truncate_for_llm(text)

    # AI 난이도 자동 조절
    adjusted = difficulty
    if prev_score is not None:
        if prev_score >= 0.85 and difficulty != "어려움":
            adjusted = {"쉬움": "보통", "보통": "어려움"}.get(difficulty, difficulty)
        elif prev_score <= 0.4 and difficulty != "쉬움":
            adjusted = {"어려움": "보통", "보통": "쉬움"}.get(difficulty, difficulty)

    system = ("당신은 한국어 시험 출제 전문가다. 강의자료에 명시된 내용만 "
              "근거로 4지선다 객관식을 출제한다. 강의자료 외 지식 사용 금지. "
              "JSON으로만 응답하라.")
    user = f"""아래 강의자료로 객관식 {num}문제를 출제하라.

[강의자료]
{text}

[조건]
- 난이도: {adjusted}
- 유형: {qtype}
- 4지선다, 정답 1개, 오답 3개
- 오답은 강의자료 내 다른 개념으로 그럴듯하게
- 중복 금지, 정답 위치 무작위 분포

[JSON 스키마]
{{
  "questions":[
    {{"question":"문제","options":["a","b","c","d"],
      "answer":"정답(options 중 하나와 정확히 일치)",
      "explanation":"상세 해설","intent":"출제 의도",
      "concept":"관련 개념(1∼3 단어)"}}
  ]
}}
"""
    data = chat_json(system, user, 0.6)
    return postprocess_questions(data.get("questions", []), num)


def postprocess_questions(questions, target):
    """중복 제거 + 정답 위치 랜덤화 + 유효성 검증."""
    cleaned, seen = [], set()
    for q in questions:
        if not all(k in q for k in ("question", "options", "answer")):
            continue
        if len(q["options"]) != 4 or q["answer"] not in q["options"]:
            continue
        h = hashlib.md5(q["question"].strip().encode()).hexdigest()
        if h in seen:
            continue
        seen.add(h)
        answer = q["answer"]
        random.shuffle(q["options"])
        q["answer"] = answer
        q.setdefault("explanation", "")
        q.setdefault("intent", "")
        q.setdefault("concept", "기타")
        cleaned.append(q)
    return cleaned[:target]


def learning_report(attempts):
    payload = [{"concept": a["question"].get("concept", "기타"),
                "intent": a["question"].get("intent", ""),
                "correct": a["is_correct"]} for a in attempts]
    system = ("당신은 학습 분석 전문가다. 결과를 분석하여 한국어 학습 리포트를 "
              "작성하라. JSON으로만 응답하라.")
    user = f"""[응시 데이터]
{json.dumps(payload, ensure_ascii=False)}

[JSON 스키마]
{{
  "weak_concepts":[], "strong_concepts":[],
  "low_understanding_topics":[], "recommendations":[],
  "review_priority":[], "study_plan":[]
}}
"""
    try:
        return chat_json(system, user, 0.4)
    except Exception:
        return {"weak_concepts": [], "strong_concepts": [],
                "low_understanding_topics": [], "recommendations": [],
                "review_priority": [], "study_plan": []}


# =========================================================================
# 5. 분석 & 시각화
# =========================================================================
def compute_stats(attempts):
    total = len(attempts)
    correct = sum(1 for a in attempts if a["is_correct"])
    return {"total": total, "correct": correct, "wrong": total - correct,
            "accuracy": round(correct / total * 100, 1) if total else 0,
            "error_rate": round((total - correct) / total * 100, 1) if total else 0}


def by_concept_df(attempts):
    from collections import defaultdict
    agg = defaultdict(lambda: {"correct": 0, "total": 0})
    for a in attempts:
        c = a["question"].get("concept", "기타")
        agg[c]["total"] += 1
        agg[c]["correct"] += int(a["is_correct"])
    rows = [{"개념": k, "정답률(%)": round(v["correct"] / v["total"] * 100, 1),
             "문제수": v["total"]} for k, v in agg.items()]
    return pd.DataFrame(rows).sort_values("정답률(%)")


def donut_chart(stats):
    fig = go.Figure(data=[go.Pie(
        labels=["정답", "오답"], values=[stats["correct"], stats["wrong"]],
        hole=0.65, marker=dict(colors=["#7C5CFF", "#FF5C7C"]),
        textinfo="label+percent")])
    fig.update_layout(showlegend=False, height=320,
                      margin=dict(t=10, b=10, l=10, r=10),
                      annotations=[dict(text=f"<b>{stats['accuracy']}%</b>",
                                        x=0.5, y=0.5, font_size=24,
                                        showarrow=False)])
    return fig


def concept_bar(df):
    return alt.Chart(df).mark_bar(cornerRadiusEnd=6).encode(
        x=alt.X("정답률(%):Q", scale=alt.Scale(domain=[0, 100])),
        y=alt.Y("개념:N", sort="-x"),
        color=alt.Color("정답률(%):Q",
                        scale=alt.Scale(scheme="purpleblue")),
        tooltip=["개념", "정답률(%)", "문제수"]
    ).properties(height=max(180, 28 * len(df)))


# =========================================================================
# 6. 내보내기 (CSV / PDF)
# =========================================================================
def wrong_notes_csv(attempts):
    wrong = [a for a in attempts if not a["is_correct"]]
    rows = [{"문제": a["question"]["question"], "내 답": a["user_answer"],
             "정답": a["question"]["answer"],
             "해설": a["question"].get("explanation", ""),
             "관련개념": a["question"].get("concept", "")} for a in wrong]
    df = pd.DataFrame(rows)
    buf = io.StringIO()
    df.to_csv(buf, index=False, encoding="utf-8-sig")
    return buf.getvalue().encode("utf-8-sig")


def wrong_notes_pdf(attempts):
    from fpdf import FPDF
    wrong = [a for a in attempts if not a["is_correct"]]
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("helvetica", "B", 14)
    pdf.cell(0, 10, "ExamForge AI - Wrong Notes", ln=True, align="C")
    pdf.set_font("helvetica", "", 11)

    def safe(t): return (t or "").encode("latin-1", "replace").decode("latin-1")

    if not wrong:
        pdf.multi_cell(0, 8, "No wrong answers!")
    for i, a in enumerate(wrong, 1):
        q = a["question"]
        pdf.set_font("helvetica", "B", 12)
        pdf.multi_cell(0, 7, safe(f"Q{i}. {q['question']}"))
        pdf.set_font("helvetica", "", 11)
        pdf.multi_cell(0, 6, safe(f"My: {a['user_answer']}"))
        pdf.multi_cell(0, 6, safe(f"Ans: {q['answer']}"))
        pdf.multi_cell(0, 6, safe(f"Concept: {q.get('concept','')}"))
        pdf.multi_cell(0, 6, safe(f"Explain: {q.get('explanation','')}"))
        pdf.ln(3)
    return bytes(pdf.output(dest="S"))


# =========================================================================
# 7. 시험 로직
# =========================================================================
def start_quiz(questions):
    return {"questions": questions, "current": 0, "attempts": [],
            "submitted": False, "selected": None, "finished": False}


def submit_answer(state, choice):
    q = state["questions"][state["current"]]
    is_correct = (choice == q["answer"])
    state["attempts"].append({"question": q, "user_answer": choice,
                              "is_correct": is_correct})
    state["submitted"] = True
    state["selected"] = choice
    return is_correct


def next_q(state):
    state["current"] += 1
    state["submitted"] = False
    state["selected"] = None
    if state["current"] >= len(state["questions"]):
        state["finished"] = True


def score_of(state):
    correct = sum(1 for a in state["attempts"] if a["is_correct"])
    return correct, len(state["questions"])


def build_retest(attempts, mode, all_qs=None):
    if mode == "wrong":
        return [a["question"] for a in attempts if not a["is_correct"]]
    if mode == "hard":
        hard = [a["question"] for a in attempts if not a["is_correct"]]
        return hard or [a["question"] for a in attempts]
    if mode == "random" and all_qs:
        pool = list(all_qs); random.shuffle(pool); return pool
    return [a["question"] for a in attempts]


# =========================================================================
# 8. UI 헬퍼
# =========================================================================
def stat_card(label, value, color="purple"):
    c = "#7C5CFF" if color == "purple" else "#22c55e" if color == "green" else "#ef4444"
    st.markdown(f"""<div class="card" style="text-align:center;">
        <div class="stat-num" style="color:{c};">{value}</div>
        <div class="stat-label">{label}</div></div>""", unsafe_allow_html=True)


def progress_header(current, total, score):
    pct = int(current / total * 100) if total else 0
    c1, c2, c3 = st.columns([2, 1, 1])
    with c1:
        st.markdown(f"**진행률** `{current}/{total}`")
        st.progress(min(pct, 100) / 100)
    with c2: st.metric("남은 문제", total - current)
    with c3: st.metric("현재 점수", score)


# =========================================================================
# 9. 페이지 렌더링
# =========================================================================
def page_home():
    st.markdown("# 🎓 ExamForge AI")
    st.caption("AI 기반 시험 문제 생성 & 학습 플랫폼")
    st.markdown("""<div class="card">
    <h3>🚀 시작하기</h3>
    <ol>
        <li><b>PDF 업로드</b> — 강의자료를 업로드</li>
        <li><b>문제 생성</b> — 난이도/유형/문항 수 선택</li>
        <li><b>시험 응시</b> — 즉시 채점 + 상세 해설</li>
        <li><b>학습 리포트</b> — 취약 개념 + 맞춤 추천</li>
        <li><b>오답노트 & 재시험</b> — 약점 보강</li>
    </ol></div>""", unsafe_allow_html=True)

    hist = fetch_history()
    if hist:
        st.subheader("📊 누적 학습 현황")
        avg = round(sum(h["score"]/h["total"]*100 for h in hist)/len(hist), 1)
        c1, c2, c3 = st.columns(3)
        with c1: stat_card("총 시험 횟수", len(hist))
        with c2: stat_card("평균 정답률", f"{avg}%", "green")
        with c3: stat_card("푼 문제 수", sum(h["total"] for h in hist))
    else:
        st.info("👈 사이드바의 **PDF 업로드**로 시작하세요.")


def page_upload():
    st.header("📄 강의자료 업로드")
    file = st.file_uploader("PDF 파일 선택 (최대 50MB)", type=["pdf"])

    if file:
        try:
            with st.spinner("📖 PDF 텍스트 추출 중..."):
                text = extract_pdf_text(file.getvalue())
            if len(text) < 200:
                st.warning("⚠️ 추출 텍스트가 너무 짧습니다. 스캔본일 수 있어요.")
                return
            st.session_state.pdf_text = text
            st.session_state.pdf_name = file.name
            st.success(f"✅ 추출 완료! ({len(text):,}자)")

            with st.expander("🔍 미리보기 (앞 1000자)"):
                st.text(text[:1000])

            st.divider()
            st.subheader("⚙️ 문제 생성 옵션")
            c1, c2, c3 = st.columns(3)
            with c1: n = st.selectbox("문제 수", QUESTION_COUNTS, index=1)
            with c2: diff = st.selectbox("난이도", DIFFICULTIES, index=1)
            with c3: qtype = st.selectbox("유형", QUESTION_TYPES, index=0)

            st.session_state.quiz_options = {"num": n, "difficulty": diff,
                                             "qtype": qtype}
            if st.button("📝 강의자료 요약 보기 →", type="primary",
                         use_container_width=True):
                st.session_state.page = "summary"
                st.rerun()
        except Exception as e:
            st.error(f"❌ 오류: {e}")


def page_summary():
    if "pdf_text" not in st.session_state:
        st.warning("먼저 PDF를 업로드하세요."); return
    st.header("📚 강의자료 핵심 요약")

    with st.spinner("AI가 요약 중..."):
        try:
            s = summarize_material(st.session_state.pdf_text)
        except Exception as e:
            st.error(f"요약 실패: {e}"); return

    st.markdown(f"""<div class="card"><h4>📌 핵심 요약</h4>
        <p>{s.get('summary','')}</p></div>""", unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### 🔑 핵심 키워드")
        st.markdown(" ".join(
            f'<span class="badge badge-purple">{k}</span>'
            for k in s.get("keywords", [])), unsafe_allow_html=True)
    with c2:
        st.markdown("#### 🏆 중요 개념 TOP 10")
        for i, c in enumerate(s.get("top_concepts", [])[:10], 1):
            st.markdown(f"**{i}. {c.get('concept','')}** — {c.get('description','')}")

    st.divider()
    opts = st.session_state.quiz_options
    st.info(f"📝 {opts['num']}문제 · **{opts['difficulty']}** · **{opts['qtype']}**")

    if st.button("🎯 시험 시작!", type="primary", use_container_width=True):
        with st.spinner("AI가 문제 생성 중... (10∼30초)"):
            try:
                prev = st.session_state.get("last_accuracy")
                qs = generate_questions(st.session_state.pdf_text,
                                        opts["num"], opts["difficulty"],
                                        opts["qtype"], prev_score=prev)
                if len(qs) < max(3, opts["num"] // 2):
                    st.error("문제 품질이 낮습니다. 다시 시도해 주세요."); return
                st.session_state.quiz = start_quiz(qs)
                st.session_state.session_saved = False
                st.session_state.page = "quiz"
                st.rerun()
            except Exception as e:
                st.error(f"생성 실패: {e}")


def page_quiz():
    quiz = st.session_state.get("quiz")
    if not quiz:
        st.warning("진행 중인 시험이 없습니다."); return
    if quiz["finished"]:
        st.session_state.page = "result"; st.rerun()

    correct, total = score_of(quiz)
    progress_header(quiz["current"], len(quiz["questions"]), correct)

    q = quiz["questions"][quiz["current"]]
    st.markdown(f"""<div class="card">
        <div class="badge badge-purple">Q{quiz['current']+1}</div>
        <div class="badge badge-purple">{q.get('concept','')}</div>
        <div class="q-title" style="margin-top:12px;">{q['question']}</div>
        </div>""", unsafe_allow_html=True)

    if not quiz["submitted"]:
        for i, opt in enumerate(q["options"]):
            if st.button(f"{chr(65+i)}.  {opt}",
                         key=f"opt_{quiz['current']}_{i}",
                         use_container_width=True):
                if submit_answer(quiz, opt):
                    st.balloons()
                st.rerun()
    else:
        last = quiz["attempts"][-1]
        if last["is_correct"]:
            st.success(f"✅ 정답입니다! — **{q['answer']}**")
        else:
            st.error(f"❌ 오답. 선택: **{last['user_answer']}** / 정답: **{q['answer']}**")
        st.markdown(f"""<div class="card">
        <b>🎯 출제 의도</b><br>{q.get('intent','')}<hr style="border-color:#2A3245;">
        <b>📖 상세 해설</b><br>{q.get('explanation','')}<hr style="border-color:#2A3245;">
        <b>🔖 관련 개념</b> &nbsp;
        <span class="badge badge-purple">{q.get('concept','')}</span></div>""",
        unsafe_allow_html=True)

        is_last = quiz["current"] + 1 >= len(quiz["questions"])
        if st.button("결과 보기 →" if is_last else "다음 문제 →",
                     type="primary", use_container_width=True):
            next_q(quiz); st.rerun()


def page_result():
    quiz = st.session_state.get("quiz")
    if not quiz or not quiz["attempts"]:
        st.warning("결과가 없습니다."); return

    correct, total = score_of(quiz)
    stats = compute_stats(quiz["attempts"])

    if not st.session_state.get("session_saved"):
        opts = st.session_state.get("quiz_options", {})
        save_session(st.session_state.get("pdf_name", "unknown.pdf"),
                     opts.get("difficulty", ""), opts.get("qtype", ""),
                     quiz["attempts"], correct, total)
        st.session_state.session_saved = True
        st.session_state.last_accuracy = correct / total

    st.header("🏁 시험 결과")
    c1, c2, c3, c4 = st.columns(4)
    with c1: stat_card("총점", f"{stats['correct']}/{stats['total']}")
    with c2: stat_card("정답률", f"{stats['accuracy']}%", "green")
    with c3: stat_card("오답률", f"{stats['error_rate']}%", "red")
    with c4: stat_card("틀린 문제", stats["wrong"], "red")

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("🎯 정답률")
        st.plotly_chart(donut_chart(stats), use_container_width=True)
    with c2:
        st.subheader("📊 개념별 성적")
        df = by_concept_df(quiz["attempts"])
        if not df.empty:
            st.altair_chart(concept_bar(df), use_container_width=True)

    st.subheader("🤖 AI 학습 리포트")
    with st.spinner("리포트 생성 중..."):
        r = learning_report(quiz["attempts"])
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("##### 🔴 취약 개념")
        for w in r.get("weak_concepts", []):
            st.markdown(f'<span class="badge badge-red">{w}</span>',
                        unsafe_allow_html=True)
        st.markdown("##### 📉 이해도 낮은 주제")
        for t in r.get("low_understanding_topics", []):
            st.write(f"- {t}")
    with c2:
        st.markdown("##### 🟢 강점")
        for s in r.get("strong_concepts", []):
            st.markdown(f'<span class="badge badge-green">{s}</span>',
                        unsafe_allow_html=True)
        st.markdown("##### 📚 추천 학습")
        for rec in r.get("recommendations", []):
            st.write(f"- {rec}")

    with st.expander("📅 7일 학습 계획"):
        for i, d in enumerate(r.get("study_plan", []), 1):
            st.write(f"**Day {i}.** {d}")
    with st.expander("🔁 복습 우선순위"):
        for i, p in enumerate(r.get("review_priority", []), 1):
            st.write(f"{i}. {p}")

    st.subheader("📝 문제별 분석")
    for i, a in enumerate(quiz["attempts"], 1):
        q = a["question"]
        mark = "✅" if a["is_correct"] else "❌"
        with st.expander(f"{mark} Q{i}. {q['question'][:50]}..."):
            st.write(f"**내 선택:** {a['user_answer']}")
            st.write(f"**정답:** {q['answer']}")
            st.write(f"**해설:** {q.get('explanation','')}")
            st.write(f"**관련 개념:** {q.get('concept','')}")

    st.divider()
    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("📓 오답노트", use_container_width=True):
            st.session_state.page = "wrong"; st.rerun()
    with c2:
        if st.button("📊 대시보드", use_container_width=True):
            st.session_state.page = "dashboard"; st.rerun()
    with c3:
        if st.button("🔄 새 시험", type="primary", use_container_width=True):
            for k in ("quiz", "session_saved"):
                st.session_state.pop(k, None)
            st.session_state.page = "upload"; st.rerun()


def page_wrong():
    quiz = st.session_state.get("quiz")
    if not quiz:
        st.warning("시험 데이터가 없습니다."); return

    st.header("📓 오답노트")
    wrong = [a for a in quiz["attempts"] if not a["is_correct"]]
    if not wrong:
        st.success("🎉 틀린 문제가 없습니다!"); return

    st.caption(f"총 **{len(wrong)}개** 오답")
    for i, a in enumerate(wrong, 1):
        q = a["question"]
        st.markdown(f"""<div class="card">
        <div class="badge badge-red">오답 {i}</div>
        <div class="badge badge-purple">{q.get('concept','')}</div>
        <div class="q-title" style="margin-top:10px;">{q['question']}</div>
        <p>❌ <b>내 답:</b> {a['user_answer']}<br>
           ✅ <b>정답:</b> {q['answer']}</p>
        <p>📖 <b>해설:</b> {q.get('explanation','')}</p>
        </div>""", unsafe_allow_html=True)

    st.divider()
    st.subheader("💾 내보내기")
    c1, c2 = st.columns(2)
    with c1:
        st.download_button("📥 CSV 다운로드",
            data=wrong_notes_csv(quiz["attempts"]),
            file_name="wrong_notes.csv", mime="text/csv",
            use_container_width=True)
    with c2:
        st.download_button("📥 PDF 다운로드",
            data=wrong_notes_pdf(quiz["attempts"]),
            file_name="wrong_notes.pdf", mime="application/pdf",
            use_container_width=True)

    st.divider()
    st.subheader("🔁 재시험")
    mode = st.radio("방식", ["틀린 문제만", "랜덤 문제", "어려운 문제만"], horizontal=True)
    mode_map = {"틀린 문제만": "wrong", "랜덤 문제": "random",
                "어려운 문제만": "hard"}
    if st.button("🔄 재시험 시작", type="primary", use_container_width=True):
        qs = build_retest(quiz["attempts"], mode_map[mode], quiz["questions"])
        if not qs:
            st.warning("재시험 문제가 없습니다."); return
        st.session_state.quiz = start_quiz(qs)
        st.session_state.session_saved = False
        st.session_state.page = "quiz"
        st.rerun()


def page_dashboard():
    st.header("📊 학습 대시보드")
    hist = fetch_history()
    if not hist:
        st.info("아직 학습 기록이 없습니다."); return

    df = pd.DataFrame(hist)
    df["accuracy"] = df["score"] / df["total"] * 100

    c1, c2, c3, c4 = st.columns(4)
    with c1: stat_card("총 시험", len(df))
    with c2: stat_card("푼 문제", int(df["total"].sum()))
    with c3: stat_card("평균 정답률", f"{round(df['accuracy'].mean(),1)}%", "green")
    with c4: stat_card("최고 정답률", f"{round(df['accuracy'].max(),1)}%", "green")

    df_sorted = df.sort_values("created_at")
    fig = px.line(df_sorted, x="created_at", y="accuracy", markers=True,
                  title="학습 추이 (정답률 %)")
    fig.update_traces(line_color="#7C5CFF", line_width=3)
    fig.update_layout(height=320)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("📋 최근 기록")
    show = df[["created_at", "pdf_name", "difficulty", "qtype",
               "score", "total", "accuracy"]].copy()
    show.columns = ["일시", "자료", "난이도", "유형", "정답", "총문항", "정답률(%)"]
    st.dataframe(show, use_container_width=True, hide_index=True)

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("난이도별")
        st.bar_chart(df.groupby("difficulty")["accuracy"].mean())
    with c2:
        st.subheader("유형별")
        st.bar_chart(df.groupby("qtype")["accuracy"].mean())


# =========================================================================
# 10. 메인 라우터
# =========================================================================
def main():
    init_db()
    if "page" not in st.session_state:
        st.session_state.page = "home"

    with st.sidebar:
        st.markdown("# 🎓 ExamForge AI")
        st.caption("AI 시험 문제 생성 & 학습")
        st.divider()
        nav = {"🏠 홈": "home", "📄 PDF 업로드": "upload",
               "📚 자료 요약": "summary", "📝 시험 진행": "quiz",
               "🏁 결과": "result", "📓 오답노트": "wrong",
               "📊 대시보드": "dashboard"}
        for label, key in nav.items():
            btn_type = "primary" if st.session_state.page == key else "secondary"
            if st.button(label, use_container_width=True, type=btn_type):
                st.session_state.page = key
                st.rerun()
        st.divider()
        st.caption("⚙️ Powered by OpenAI")
        if st.button("🗑️ 세션 초기화", use_container_width=True):
            for k in list(st.session_state.keys()):
                if k != "page":
                    del st.session_state[k]
            st.session_state.page = "home"
            st.rerun()

    pages = {"home": page_home, "upload": page_upload,
             "summary": page_summary, "quiz": page_quiz,
             "result": page_result, "wrong": page_wrong,
             "dashboard": page_dashboard}
    try:
        pages.get(st.session_state.page, page_home)()
    except Exception as e:
        st.error("😢 예상치 못한 오류가 발생했습니다.")
        st.exception(e)


if __name__ == "__main__":
    main()
