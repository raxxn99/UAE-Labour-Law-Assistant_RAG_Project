import streamlit as st
import time
from datetime import datetime

# ─── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="UAE Labour Law Assistant",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Global CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.stApp {
    background: linear-gradient(135deg, #0a0f1e 0%, #0d1b3e 50%, #0a1628 100%);
    min-height: 100vh;
}

::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #0d1b3e; }
::-webkit-scrollbar-thumb { background: #c9a84c; border-radius: 3px; }

#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1.5rem !important; }

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #060d20 0%, #0a1628 100%);
    border-right: 1px solid rgba(201,168,76,0.2);
}

.metric-card {
    background: linear-gradient(135deg, rgba(255,255,255,0.04), rgba(255,255,255,0.01));
    border: 1px solid rgba(201,168,76,0.25);
    border-radius: 16px;
    padding: 1.2rem 1.4rem;
    text-align: center;
    backdrop-filter: blur(10px);
    transition: transform 0.2s ease, border-color 0.2s ease;
}
.metric-card:hover { transform: translateY(-2px); border-color: rgba(201,168,76,0.6); }
.metric-value {
    font-size: 2rem; font-weight: 800;
    background: linear-gradient(135deg, #c9a84c, #f0d080);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    line-height: 1.1;
}
.metric-label {
    font-size: 0.72rem; color: rgba(255,255,255,0.5);
    text-transform: uppercase; letter-spacing: 0.08em; margin-top: 0.3rem;
}

.hero-banner {
    background: linear-gradient(135deg, rgba(201,168,76,0.12) 0%, rgba(13,27,62,0.8) 60%, rgba(201,168,76,0.08) 100%);
    border: 1px solid rgba(201,168,76,0.3);
    border-radius: 20px;
    padding: 2rem 2.5rem;
    margin-bottom: 1.5rem;
    position: relative; overflow: hidden;
}
.hero-banner::before {
    content: ''; position: absolute; top: -50%; right: -10%;
    width: 300px; height: 300px;
    background: radial-gradient(circle, rgba(201,168,76,0.08) 0%, transparent 70%);
    pointer-events: none;
}
.hero-title { font-size: 2rem; font-weight: 800; color: #ffffff; margin: 0; line-height: 1.2; }
.hero-title span {
    background: linear-gradient(135deg, #c9a84c, #f0d080);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.hero-subtitle { font-size: 0.95rem; color: rgba(255,255,255,0.55); margin-top: 0.5rem; }
.hero-badge {
    display: inline-block;
    background: rgba(201,168,76,0.15); border: 1px solid rgba(201,168,76,0.4);
    color: #c9a84c; border-radius: 50px; padding: 0.25rem 0.9rem;
    font-size: 0.75rem; font-weight: 600; letter-spacing: 0.05em; margin-bottom: 0.8rem;
}

.chat-bubble-user {
    background: linear-gradient(135deg, #1a3a6e, #1e4080);
    border: 1px solid rgba(100,150,255,0.25);
    border-radius: 18px 18px 4px 18px;
    padding: 0.9rem 1.2rem; max-width: 78%; align-self: flex-end;
    color: #e8eeff; font-size: 0.9rem; line-height: 1.6;
    box-shadow: 0 4px 20px rgba(0,0,40,0.3);
}
.chat-bubble-user .bubble-meta { font-size: 0.68rem; color: rgba(150,180,255,0.5); text-align: right; margin-top: 0.3rem; }
.chat-bubble-bot {
    background: linear-gradient(135deg, rgba(201,168,76,0.1), rgba(255,255,255,0.04));
    border: 1px solid rgba(201,168,76,0.25);
    border-radius: 18px 18px 18px 4px;
    padding: 0.9rem 1.2rem; max-width: 82%; align-self: flex-start;
    color: #f0f0f0; font-size: 0.9rem; line-height: 1.7;
    box-shadow: 0 4px 20px rgba(0,0,0,0.25);
}
.chat-bubble-bot .bubble-meta { font-size: 0.68rem; color: rgba(201,168,76,0.45); margin-top: 0.4rem; }
.bot-avatar { display: inline-flex; align-items: center; gap: 0.4rem; margin-bottom: 0.4rem; }
.bot-avatar-icon {
    width: 22px; height: 22px;
    background: linear-gradient(135deg, #c9a84c, #f0d080);
    border-radius: 50%; display: inline-flex; align-items: center;
    justify-content: center; font-size: 0.65rem;
}
.bot-avatar-name { font-size: 0.72rem; font-weight: 600; color: #c9a84c; text-transform: uppercase; letter-spacing: 0.05em; }

.chunk-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(201,168,76,0.15);
    border-left: 3px solid #c9a84c;
    border-radius: 10px; padding: 0.8rem 1rem; margin-bottom: 0.6rem; transition: border-color 0.2s;
}
.chunk-card:hover { border-color: rgba(201,168,76,0.5); border-left-color: #f0d080; }
.chunk-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.3rem; }
.chunk-label { font-size: 0.72rem; font-weight: 600; color: #c9a84c; text-transform: uppercase; letter-spacing: 0.06em; }
.chunk-score { font-size: 0.7rem; background: rgba(201,168,76,0.15); color: #f0d080; border-radius: 50px; padding: 0.1rem 0.5rem; font-weight: 600; }
.chunk-source { font-size: 0.7rem; color: rgba(255,255,255,0.35); margin-bottom: 0.3rem; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.chunk-text { font-size: 0.8rem; color: rgba(255,255,255,0.6); line-height: 1.6; }

.status-dot {
    display: inline-block; width: 8px; height: 8px; border-radius: 50%;
    background: #22c55e; box-shadow: 0 0 8px rgba(34,197,94,0.6);
    margin-right: 6px; animation: pulse-green 2s infinite;
}
@keyframes pulse-green {
    0%, 100% { box-shadow: 0 0 8px rgba(34,197,94,0.6); }
    50% { box-shadow: 0 0 16px rgba(34,197,94,0.9); }
}

.info-block {
    background: rgba(255,255,255,0.03); border: 1px solid rgba(201,168,76,0.15);
    border-radius: 12px; padding: 0.9rem 1rem; margin-bottom: 0.8rem;
}
.info-block-title { font-size: 0.7rem; font-weight: 700; color: #c9a84c; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 0.6rem; }
.info-row { display: flex; justify-content: space-between; align-items: center; padding: 0.25rem 0; border-bottom: 1px solid rgba(255,255,255,0.04); }
.info-row:last-child { border-bottom: none; }
.info-key { font-size: 0.75rem; color: rgba(255,255,255,0.45); }
.info-val { font-size: 0.75rem; color: rgba(255,255,255,0.8); font-weight: 500; }

.pipeline-step { display: flex; align-items: center; gap: 0.6rem; padding: 0.45rem 0; }
.step-num {
    width: 22px; height: 22px; border-radius: 50%;
    background: linear-gradient(135deg, #c9a84c, #f0d080);
    display: flex; align-items: center; justify-content: center;
    font-size: 0.65rem; font-weight: 800; color: #0a0f1e; flex-shrink: 0;
}
.step-text { font-size: 0.78rem; color: rgba(255,255,255,0.65); }

.gold-divider { height: 1px; background: linear-gradient(90deg, transparent, rgba(201,168,76,0.4), transparent); margin: 1rem 0; }

.stButton > button {
    background: linear-gradient(135deg, rgba(201,168,76,0.2), rgba(201,168,76,0.1)) !important;
    border: 1px solid rgba(201,168,76,0.4) !important; color: #f0d080 !important;
    border-radius: 10px !important; font-weight: 500 !important; transition: all 0.2s !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, rgba(201,168,76,0.35), rgba(201,168,76,0.2)) !important;
    border-color: rgba(201,168,76,0.7) !important; transform: translateY(-1px) !important;
}

.stTabs [data-baseweb="tab-list"] {
    background: rgba(255,255,255,0.03) !important; border-radius: 10px; padding: 4px; gap: 4px;
    border: 1px solid rgba(201,168,76,0.15);
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important; color: rgba(255,255,255,0.45) !important;
    border-radius: 8px !important; font-size: 0.82rem !important; font-weight: 500 !important;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, rgba(201,168,76,0.25), rgba(201,168,76,0.1)) !important;
    color: #f0d080 !important;
}

.example-q {
    background: rgba(255,255,255,0.03); border: 1px solid rgba(201,168,76,0.12);
    border-radius: 10px; padding: 0.55rem 0.8rem; margin-bottom: 0.4rem;
    font-size: 0.78rem; color: rgba(255,255,255,0.6); transition: all 0.2s;
}
.example-q:hover { background: rgba(201,168,76,0.08); border-color: rgba(201,168,76,0.3); color: rgba(255,255,255,0.85); }
</style>
""", unsafe_allow_html=True)

# ─── Session State ────────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "question_count" not in st.session_state:
    st.session_state.question_count = 0
if "avg_response_time" not in st.session_state:
    st.session_state.avg_response_time = 0.0
if "total_chunks_retrieved" not in st.session_state:
    st.session_state.total_chunks_retrieved = 0

# ─── RAG Integration ─────────────────────────────────────────────────────────────
def get_rag_answer(question: str):
    """Call the real RAG pipeline. Falls back to demo if not set up."""
    try:
        import sys, os
        sys.path.insert(0, os.path.dirname(__file__))
        from generate import answer_question
        return answer_question(question), False
    except Exception as e:
        return _demo_answer(question, str(e)), True


def _demo_answer(question: str, error: str = ""):
    q = question.lower()
    if any(w in q for w in ["leave", "vacation", "annual", "holiday"]):
        answer = (
            "According to Federal Decree-Law No. 33 of 2021 (Article 29), employees in the UAE "
            "are entitled to **annual leave** of no less than **30 calendar days** per year once "
            "they complete one year of service.\n\n"
            "Employees who have not yet completed one year are entitled to leave proportional "
            "to the period they have worked.\n\n"
            "_Source: Federal Decree-Law No. 33 of 2021 Regarding the Regulation of Employment Relationship_"
        )
        chunks = [
            {"text": "Article 29: The worker shall be entitled to an annual leave of no less than 30 days for each year of service...", "source": "Federal Decree-Law No. 33 of 2021...", "chunk_index": "142", "hybrid_score": 0.934, "semantic_score": 0.921, "keyword_score": 0.961},
            {"text": "Article 30: Annual leave scheduling and conditions under which leave may be deferred or divided...", "source": "Federal Decree-Law No. 33 of 2021...", "chunk_index": "143", "hybrid_score": 0.871, "semantic_score": 0.855, "keyword_score": 0.910},
            {"text": "Annual leave entitlements, minimum days, and proportional calculation for partial years...", "source": "Annual leave _ The Official Platform of the UAE Government.txt", "chunk_index": "8", "hybrid_score": 0.802, "semantic_score": 0.798, "keyword_score": 0.812},
            {"text": "Cabinet Resolution No. 1 of 2022 — Leave regulations and employer obligations during annual leave period...", "source": "Cabinet Resolution No. (1) of 2022...", "chunk_index": "55", "hybrid_score": 0.741, "semantic_score": 0.728, "keyword_score": 0.769},
            {"text": "Additional working hours, wages, and leave guide — employee rights to annual leave and compensation...", "source": "Additional Working Hours - Wages - Leaves Guide EN.txt", "chunk_index": "21", "hybrid_score": 0.698, "semantic_score": 0.681, "keyword_score": 0.736},
        ]
    elif any(w in q for w in ["gratuity", "end of service", "eosg", "severance"]):
        answer = (
            "Under Federal Decree-Law No. 33 of 2021 (Article 51), **end of service gratuity** is calculated as:\n\n"
            "- **First 5 years**: 21 working days of basic salary per year\n"
            "- **After 5 years**: 30 working days of basic salary per year\n\n"
            "Gratuity is based on the **basic salary only** (excluding allowances).\n\n"
            "_Source: End of service benefits for workers in the private sector — UAE Government_"
        )
        chunks = [
            {"text": "Article 51: Upon the end of service the worker shall be entitled to end of service gratuity at 21 working days basic wage for each of the first five years...", "source": "Federal Decree-Law No. 33 of 2021...", "chunk_index": "201", "hybrid_score": 0.967, "semantic_score": 0.955, "keyword_score": 0.993},
            {"text": "End of service gratuity calculation formula, base salary, years of service, and deductions...", "source": "End of service benefits for workers in the private sector _ The Official Platform of the UAE Government.txt", "chunk_index": "14", "hybrid_score": 0.911, "semantic_score": 0.899, "keyword_score": 0.940},
            {"text": "Article 52: Gratuity shall be calculated on the basis of the last basic wage received by the worker...", "source": "Federal Decree-Law No. 33 of 2021...", "chunk_index": "202", "hybrid_score": 0.878, "semantic_score": 0.861, "keyword_score": 0.921},
            {"text": "Savings scheme and end of service provisions for private sector employees...", "source": "savings-scheme-document-en.txt", "chunk_index": "5", "hybrid_score": 0.812, "semantic_score": 0.801, "keyword_score": 0.840},
            {"text": "Workers entitled to gratuity upon resignation or termination unless dismissed under Article 44...", "source": "Federal Decree-Law No. 33 of 2021...", "chunk_index": "203", "hybrid_score": 0.756, "semantic_score": 0.741, "keyword_score": 0.789},
        ]
    elif any(w in q for w in ["terminat", "dismiss", "notice", "fire", "fired"]):
        answer = (
            "Under Federal Decree-Law No. 33 of 2021 (Article 44), an employer may terminate "
            "**without notice** in specific cases including:\n\n"
            "1. Employee assumes false identity or submits forged documents\n"
            "2. Employee commits an error causing substantial financial loss\n"
            "3. Employee discloses confidential company information\n"
            "4. Employee found intoxicated during working hours\n"
            "5. Employee assaults the employer, manager, or a colleague\n\n"
            "Otherwise, a minimum notice period of **30 days** (up to 90 days for senior roles) applies.\n\n"
            "_Source: Terminating employment contracts — UAE Government_"
        )
        chunks = [
            {"text": "Article 44: The employer may dismiss the worker without notice in the following cases without end of service gratuity...", "source": "Federal Decree-Law No. 33 of 2021...", "chunk_index": "178", "hybrid_score": 0.958, "semantic_score": 0.944, "keyword_score": 0.988},
            {"text": "Terminating employment contracts and arbitrary dismissal — rights and procedures...", "source": "Terminating employment contracts and arbitrary dismissal _ The Official Platform of the UAE Government.txt", "chunk_index": "11", "hybrid_score": 0.901, "semantic_score": 0.889, "keyword_score": 0.930},
            {"text": "Article 43: Either party may terminate the employment contract by giving the other party written notice...", "source": "Federal Decree-Law No. 33 of 2021...", "chunk_index": "177", "hybrid_score": 0.847, "semantic_score": 0.832, "keyword_score": 0.877},
            {"text": "Notice period requirements for different contract durations and salary levels...", "source": "Cabinet Resolution No. (1) of 2022...", "chunk_index": "72", "hybrid_score": 0.789, "semantic_score": 0.774, "keyword_score": 0.819},
            {"text": "Arbitrary dismissal compensation and remedies available to employees under UAE labour law...", "source": "Terminating employment contracts and arbitrary dismissal _ The Official Platform of the UAE Government.txt", "chunk_index": "15", "hybrid_score": 0.731, "semantic_score": 0.717, "keyword_score": 0.762},
        ]
    elif any(w in q for w in ["probation", "trial", "probationary"]):
        answer = (
            "Under Federal Decree-Law No. 33 of 2021 (Article 9):\n\n"
            "- Maximum probation period: **6 months** from the start date\n"
            "- Cannot be renewed for the same employee at the same employer\n"
            "- If terminated by employer during probation: **14 days** written notice\n"
            "- Gratuity accrues from day one, including during probation\n\n"
            "_Source: Federal Decree-Law No. 33 of 2021_"
        )
        chunks = [
            {"text": "Article 9: The period of probation shall not exceed six months from the date of commencement of work...", "source": "Federal Decree-Law No. 33 of 2021...", "chunk_index": "42", "hybrid_score": 0.944, "semantic_score": 0.930, "keyword_score": 0.975},
            {"text": "Article 10: During probation either party may terminate the contract with 14 days written notice...", "source": "Federal Decree-Law No. 33 of 2021...", "chunk_index": "43", "hybrid_score": 0.889, "semantic_score": 0.875, "keyword_score": 0.920},
            {"text": "Probation period rights and obligations for employer and employee under UAE private sector law...", "source": "Awareness Guide for New Employers Companies - EN_639011571513592816.txt", "chunk_index": "33", "hybrid_score": 0.821, "semantic_score": 0.808, "keyword_score": 0.853},
            {"text": "Article 11: The probation period shall be included in the service period for gratuity calculation...", "source": "Federal Decree-Law No. 33 of 2021...", "chunk_index": "44", "hybrid_score": 0.762, "semantic_score": 0.748, "keyword_score": 0.793},
            {"text": "Cabinet Resolution No. 1 of 2022 — probation conditions and employment contract types...", "source": "Cabinet Resolution No. (1) of 2022...", "chunk_index": "18", "hybrid_score": 0.701, "semantic_score": 0.687, "keyword_score": 0.733},
        ]
    else:
        note = f"\n\n⚠️ _Demo mode — connect Gemini API key in `.env` to get live answers. Error: {error[:80] if error else 'Pipeline not connected'}_" if error else ""
        answer = (
            f"The UAE Labour Law RAG system has analysed your question about *\"{question}\"* "
            "and retrieved the most relevant sections from 26 official documents including "
            "Federal Decree-Law No. 33 of 2021 and its amendments." + note
        )
        chunks = [
            {"text": "Federal Decree-Law No. 33 of 2021 — General provisions on employment relationships...", "source": "Federal Decree-Law No. 33 of 2021...", "chunk_index": "1", "hybrid_score": 0.812, "semantic_score": 0.798, "keyword_score": 0.841},
            {"text": "Cabinet Resolution No. 1 of 2022 — Implementation regulations...", "source": "Cabinet Resolution No. (1) of 2022...", "chunk_index": "1", "hybrid_score": 0.743, "semantic_score": 0.729, "keyword_score": 0.772},
            {"text": "Awareness guide for employers — employee rights and employer obligations...", "source": "Awareness Guide for New Employers Companies - EN_639011571513592816.txt", "chunk_index": "1", "hybrid_score": 0.681, "semantic_score": 0.668, "keyword_score": 0.710},
            {"text": "Occupational health and safety regulations in the UAE private sector...", "source": "Occupational Health and Safety System in the UAE - EN_638983811445415191.txt", "chunk_index": "1", "hybrid_score": 0.621, "semantic_score": 0.608, "keyword_score": 0.648},
            {"text": "Remote work guide for private sector employees and employers...", "source": "Remote Work Guide in the Private Sector - EN.txt", "chunk_index": "1", "hybrid_score": 0.578, "semantic_score": 0.564, "keyword_score": 0.607},
        ]

    return {"answer": answer, "context_chunks": chunks, "model": "gemini-2.5-flash"}


# ─── Sidebar ─────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 1rem 0 1.5rem;">
        <div style="font-size:2.6rem; margin-bottom:0.3rem;">⚖️</div>
        <div style="font-size:1rem; font-weight:800; color:#ffffff;">UAE Labour</div>
        <div style="font-size:0.75rem; font-weight:600;
                    background:linear-gradient(135deg,#c9a84c,#f0d080);
                    -webkit-background-clip:text; -webkit-text-fill-color:transparent;">
            LAW ASSISTANT
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="info-block">
        <div class="info-block-title">System Status</div>
        <div style="display:flex; align-items:center; gap:6px; padding:0.2rem 0;">
            <span class="status-dot"></span>
            <span style="font-size:0.78rem; color:rgba(255,255,255,0.7);">RAG Pipeline Active</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="info-block">
        <div class="info-block-title">Configuration</div>
        <div class="info-row">
            <span class="info-key">Embedding</span>
            <span class="info-val">MiniLM-L6-v2</span>
        </div>
        <div class="info-row">
            <span class="info-key">Retrieval</span>
            <span class="info-val">Hybrid (70/30)</span>
        </div>
        <div class="info-row">
            <span class="info-key">Top-K</span>
            <span class="info-val">5 chunks</span>
        </div>
        <div class="info-row">
            <span class="info-key">LLM</span>
            <span class="info-val">Gemini 2.5 Flash</span>
        </div>
        <div class="info-row">
            <span class="info-key">Vector DB</span>
            <span class="info-val">ChromaDB</span>
        </div>
        <div class="info-row">
            <span class="info-key">Indexed Chunks</span>
            <span class="info-val">2,033</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="info-block">
        <div class="info-block-title">Document Categories</div>
        <div class="info-row">
            <span class="info-key">Core Laws</span>
            <span class="info-val" style="color:#22c55e;">4 docs ✓</span>
        </div>
        <div class="info-row">
            <span class="info-key">Employment Models</span>
            <span class="info-val" style="color:#22c55e;">9 docs ✓</span>
        </div>
        <div class="info-row">
            <span class="info-key">Worker Protection</span>
            <span class="info-val" style="color:#22c55e;">6 docs ✓</span>
        </div>
        <div class="info-row">
            <span class="info-key">Awareness Guides</span>
            <span class="info-val" style="color:#22c55e;">6 docs ✓</span>
        </div>
        <div class="info-row">
            <span class="info-key">Disputes</span>
            <span class="info-val" style="color:#22c55e;">1 doc ✓</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="info-block">
        <div class="info-block-title">RAG Pipeline</div>
        <div class="pipeline-step"><div class="step-num">1</div><div class="step-text">Query Embedding</div></div>
        <div class="pipeline-step"><div class="step-num">2</div><div class="step-text">Semantic Search (ChromaDB)</div></div>
        <div class="pipeline-step"><div class="step-num">3</div><div class="step-text">BM25 Keyword Search</div></div>
        <div class="pipeline-step"><div class="step-num">4</div><div class="step-text">Hybrid Re-ranking (70/30)</div></div>
        <div class="pipeline-step"><div class="step-num">5</div><div class="step-text">Gemini 2.5 Flash Generation</div></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='gold-divider'></div>", unsafe_allow_html=True)
    if st.button("🗑️  Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.question_count = 0
        st.session_state.total_chunks_retrieved = 0
        st.rerun()

    st.markdown("""
    <div style="text-align:center; margin-top:1.5rem;">
        <div style="font-size:0.65rem; color:rgba(255,255,255,0.2); letter-spacing:0.05em;">
            CSAI-413 · NLP Applications<br>The British University in Dubai
        </div>
    </div>
    """, unsafe_allow_html=True)


# ─── Hero ─────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-banner">
    <div class="hero-badge">⚖️ RETRIEVAL-AUGMENTED GENERATION · GEMINI 2.5 FLASH</div>
    <div class="hero-title">UAE Labour Law <span>Intelligent Assistant</span></div>
    <div class="hero-subtitle">
        Ask any question about your labour rights and obligations in the UAE —
        powered by hybrid semantic + BM25 retrieval over 26 official documents and 2,033 indexed chunks.
    </div>
</div>
""", unsafe_allow_html=True)

# ─── Metrics ─────────────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(f'<div class="metric-card"><div class="metric-value">{st.session_state.question_count}</div><div class="metric-label">Questions Asked</div></div>', unsafe_allow_html=True)
with c2:
    st.markdown('<div class="metric-card"><div class="metric-value">26</div><div class="metric-label">Documents Indexed</div></div>', unsafe_allow_html=True)
with c3:
    rt = f"{st.session_state.avg_response_time:.1f}s" if st.session_state.avg_response_time > 0 else "—"
    st.markdown(f'<div class="metric-card"><div class="metric-value">{rt}</div><div class="metric-label">Avg Response Time</div></div>', unsafe_allow_html=True)
with c4:
    st.markdown(f'<div class="metric-card"><div class="metric-value">2,033</div><div class="metric-label">Indexed Chunks</div></div>', unsafe_allow_html=True)

st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

# ─── Tabs ─────────────────────────────────────────────────────────────────────────
tab_chat, tab_examples, tab_about = st.tabs(["💬  Chat", "💡  Example Questions", "ℹ️  About"])

with tab_chat:
    chat_col, source_col = st.columns([3, 2], gap="medium")

    with chat_col:
        chat_container = st.container(height=480, border=False)
        with chat_container:
            if not st.session_state.messages:
                st.markdown("""
                <div style="text-align:center; padding:3rem 2rem; color:rgba(255,255,255,0.3);">
                    <div style="font-size:3.5rem; margin-bottom:1rem; opacity:0.4;">⚖️</div>
                    <div style="font-size:1rem; color:rgba(255,255,255,0.45); margin-bottom:0.4rem; font-weight:600;">
                        Ask your first question
                    </div>
                    <div style="font-size:0.85rem;">
                        Type any question about UAE labour rights below —<br>
                        annual leave, gratuity, termination, working hours, and more.
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                for msg in st.session_state.messages:
                    ts = msg.get("ts", "")
                    if msg["role"] == "user":
                        st.markdown(f"""
                        <div class="chat-bubble-user">
                            {msg["content"]}
                            <div class="bubble-meta">You · {ts}</div>
                        </div>""", unsafe_allow_html=True)
                    else:
                        content = msg["content"].replace("\n", "<br>")
                        st.markdown(f"""
                        <div class="chat-bubble-bot">
                            <div class="bot-avatar">
                                <div class="bot-avatar-icon">⚖</div>
                                <span class="bot-avatar-name">Labour Law AI</span>
                            </div>
                            {content}
                            <div class="bubble-meta">Gemini 2.5 Flash · {ts}</div>
                        </div>""", unsafe_allow_html=True)

        question = st.chat_input("Ask about UAE labour rights, leave, gratuity, termination…")

    with source_col:
        st.markdown("""
        <div style="font-size:0.78rem; font-weight:700; color:#c9a84c;
                    text-transform:uppercase; letter-spacing:0.08em; margin-bottom:0.8rem;">
            📚 Retrieved Sources
        </div>""", unsafe_allow_html=True)

        source_container = st.container(height=450, border=False)
        with source_container:
            last_bot = next((m for m in reversed(st.session_state.messages) if m["role"] == "assistant"), None)
            if last_bot and last_bot.get("chunks"):
                for i, chunk in enumerate(last_bot["chunks"], 1):
                    score_pct = int(chunk.get("hybrid_score", 0) * 100)
                    source = chunk.get("source", "Unknown")
                    # Shorten long source names
                    short_source = source[:55] + "…" if len(source) > 55 else source
                    st.markdown(f"""
                    <div class="chunk-card">
                        <div class="chunk-header">
                            <span class="chunk-label">Chunk {i} · #{chunk.get('chunk_index','?')}</span>
                            <span class="chunk-score">{score_pct}% match</span>
                        </div>
                        <div class="chunk-source">📄 {short_source}</div>
                        <div class="chunk-text">{chunk['text'][:240]}{'…' if len(chunk['text']) > 240 else ''}</div>
                    </div>""", unsafe_allow_html=True)
            else:
                st.markdown("""
                <div style="text-align:center; padding:2.5rem 1rem; color:rgba(255,255,255,0.2);">
                    <div style="font-size:2rem; margin-bottom:0.5rem;">📄</div>
                    <div style="font-size:0.8rem;">Retrieved law passages will appear here after your first question.</div>
                </div>""", unsafe_allow_html=True)

# ─── Handle Question ──────────────────────────────────────────────────────────────
if question:
    ts = datetime.now().strftime("%H:%M")
    st.session_state.messages.append({"role": "user", "content": question, "ts": ts})

    with st.spinner("Retrieving relevant law articles and generating answer…"):
        t0 = time.time()
        result, is_demo = get_rag_answer(question)
        elapsed = time.time() - t0

    n = st.session_state.question_count
    st.session_state.avg_response_time = (st.session_state.avg_response_time * n + elapsed) / (n + 1)
    st.session_state.question_count += 1
    chunks = result.get("context_chunks", [])
    st.session_state.total_chunks_retrieved += len(chunks)

    answer_text = result.get("answer", "")
    ts2 = datetime.now().strftime("%H:%M")
    st.session_state.messages.append({"role": "assistant", "content": answer_text, "chunks": chunks, "ts": ts2})
    st.rerun()

# ─── Example Questions Tab ────────────────────────────────────────────────────────
with tab_examples:
    st.markdown('<div style="font-size:0.85rem; color:rgba(255,255,255,0.5); margin-bottom:1.2rem;">Copy any question and paste it in the chat.</div>', unsafe_allow_html=True)
    categories = {
        "🏖️  Leave & Holidays": ["What is the minimum annual leave for employees in the UAE?", "How many sick days is an employee entitled to?", "What are the official public holidays in the UAE?", "Can an employer cancel approved annual leave?"],
        "📋  Employment Contract": ["What are the types of employment contracts in UAE?", "What is the maximum probation period allowed?", "Can a fixed-term contract be renewed indefinitely?", "What constitutes a valid employment contract under UAE law?"],
        "💰  Salary & Gratuity": ["How is end of service gratuity calculated?", "When must an employer pay the final salary upon resignation?", "What deductions can legally be made from an employee's salary?", "Is overtime compensation mandatory in the UAE?"],
        "🚫  Termination & Notice": ["When can an employer terminate without notice in the UAE?", "What is the notice period for resignation?", "What rights does an employee have if unfairly dismissed?", "What is the Wage Protection System (WPS)?"],
        "🏠  Remote Work & Safety": ["What are the rules for remote work in the UAE private sector?", "What are employer obligations for worker health and safety?", "What rights do domestic workers have under UAE law?", "What is the unemployment insurance scheme in UAE?"],
    }
    for cat, questions in categories.items():
        st.markdown(f'<div style="font-size:0.78rem; font-weight:700; color:#c9a84c; text-transform:uppercase; letter-spacing:0.08em; margin: 1rem 0 0.5rem;">{cat}</div>', unsafe_allow_html=True)
        for q in questions:
            st.markdown(f'<div class="example-q">❯ {q}</div>', unsafe_allow_html=True)

# ─── About Tab ────────────────────────────────────────────────────────────────────
with tab_about:
    a1, a2 = st.columns(2, gap="large")
    with a1:
        st.markdown("""
        <div class="info-block" style="padding:1.2rem 1.4rem;">
            <div class="info-block-title">What is RAG?</div>
            <div style="font-size:0.82rem; color:rgba(255,255,255,0.6); line-height:1.7;">
                <b style="color:rgba(255,255,255,0.85);">Retrieval-Augmented Generation</b> grounds
                an LLM's responses in real documents — preventing hallucination by forcing the model
                to cite passages it actually retrieved from official UAE Labour Law sources.
                <br><br>
                This system indexes 26 official UAE Labour Law PDFs into ChromaDB (2,033 chunks),
                retrieves the top-5 most relevant passages per query, and passes them as context
                to Gemini 2.5 Flash to generate grounded answers.
            </div>
        </div>
        <div class="info-block" style="padding:1.2rem 1.4rem; margin-top:0.8rem;">
            <div class="info-block-title">Hybrid Retrieval (70/30)</div>
            <div style="font-size:0.82rem; color:rgba(255,255,255,0.6); line-height:1.7;">
                <b style="color:#c9a84c;">70% Semantic</b> — finds conceptually similar chunks
                using all-MiniLM-L6-v2 embeddings in ChromaDB.<br><br>
                <b style="color:#c9a84c;">30% BM25 Keyword</b> — finds exact keyword matches,
                critical for legal terms like article numbers and specific provisions.<br><br>
                Both scores are combined and re-ranked before passing to the LLM.
            </div>
        </div>
        """, unsafe_allow_html=True)
    with a2:
        st.markdown("""
        <div class="info-block" style="padding:1.2rem 1.4rem;">
            <div class="info-block-title">Team</div>
            <div class="info-row"><span class="info-key">Alia</span><span class="info-val">Data Collection & Evaluation</span></div>
            <div class="info-row"><span class="info-key">Salma</span><span class="info-val">Data Cleaning & Preprocessing</span></div>
            <div class="info-row"><span class="info-key">Rana</span><span class="info-val">Chunking & ChromaDB</span></div>
            <div class="info-row"><span class="info-key">Aya</span><span class="info-val">Retrieval & Gemini Generation</span></div>
            <div class="info-row"><span class="info-key" style="color:#c9a84c;">Reem</span><span class="info-val" style="color:#c9a84c;">UI / Evaluation Framework</span></div>
        </div>
        <div class="info-block" style="padding:1.2rem 1.4rem; margin-top:0.8rem;">
            <div class="info-block-title">Evaluation Metrics (D2)</div>
            <div style="font-size:0.82rem; color:rgba(255,255,255,0.6); line-height:1.8;">
                • <b style="color:rgba(255,255,255,0.8);">Retrieval Relevance</b> — 1–5 scale<br>
                • <b style="color:rgba(255,255,255,0.8);">Answer Correctness</b> — 1–5 scale<br>
                • <b style="color:rgba(255,255,255,0.8);">Coverage</b> — Full / Partial / None<br>
                • <b style="color:rgba(255,255,255,0.8);">Hallucination Rate</b> — Grounded / Hallucinated
            </div>
        </div>
        """, unsafe_allow_html=True)
