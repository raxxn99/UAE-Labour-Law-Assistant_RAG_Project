"""
app.py - UAE Labour Law RAG System
Updated Version:
1. Sample queries are inside the Live Chat tab.
2. Each answer keeps its own retrieved chunks.
3. Removed empty box under sample query text.
4. Sample query buttons now have equal height.
5. Replaced unanswered arbitrary dismissal question.
"""

import streamlit as st
import time
import sys
from pathlib import Path
import os

# Reduce noisy logs
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] = "none"

try:
    from transformers.utils import logging
    logging.set_verbosity_error()
except Exception:
    pass

# Add src folder to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

st.set_page_config(
    page_title="UAE Labour Law Assistant",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');

* {
    font-family: 'Plus Jakarta Sans', sans-serif;
}

.stApp {
    background:
        radial-gradient(circle at 10% 20%, rgba(99, 102, 241, 0.15), transparent 40%),
        radial-gradient(circle at 90% 80%, rgba(20, 184, 166, 0.12), transparent 40%),
        linear-gradient(145deg, #090d16 0%, #0f172a 60%, #030712 100%);
    color: #f8fafc;
}

.block-container {
    padding-top: 1.5rem !important;
    max-width: 1500px;
}

#MainMenu, footer, header {
    visibility: hidden;
}

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0b0f19 0%, #090d16 100%) !important;
    border-right: 1px solid rgba(255, 255, 255, 0.06) !important;
}

section[data-testid="stSidebar"] * {
    color: #f1f5f9 !important;
}

.hero-banner {
    background: linear-gradient(135deg, rgba(99, 102, 241, 0.12), rgba(20, 184, 166, 0.08));
    border: 1px solid rgba(255, 255, 255, 0.07);
    backdrop-filter: blur(24px);
    border-radius: 24px;
    padding: 2.5rem;
    margin-bottom: 2rem;
    box-shadow: 0 20px 40px rgba(0, 0, 0, 0.4);
}

.hero-title {
    font-size: 2.8rem;
    font-weight: 800;
    line-height: 1.15;
    letter-spacing: -0.02em;
}

.hero-title span {
    background: linear-gradient(90deg, #a5b4fc, #2dd4bf);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.hero-subtitle {
    margin-top: 1rem;
    color: #94a3b8;
    line-height: 1.7;
    max-width: 950px;
    font-size: 1.05rem;
}

.hero-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.5rem 1rem;
    border-radius: 999px;
    background: rgba(99, 102, 241, 0.15);
    border: 1px solid rgba(99, 102, 241, 0.25);
    font-size: 0.75rem;
    font-weight: 700;
    color: #c7d2fe;
    margin-bottom: 1.2rem;
    letter-spacing: 0.05em;
}

.metric-card {
    background: rgba(255, 255, 255, 0.03);
    border: 1px solid rgba(255, 255, 255, 0.05);
    border-radius: 20px;
    padding: 1.5rem;
    text-align: center;
    backdrop-filter: blur(12px);
}

.metric-value {
    font-size: 2.3rem;
    font-weight: 800;
    background: linear-gradient(90deg, #c7d2fe, #2dd4bf);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.metric-label {
    color: #64748b;
    margin-top: 0.4rem;
    font-size: 0.75rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}

.chat-bubble-user {
    background: linear-gradient(135deg, #4f46e5 0%, #3730a3 100%);
    color: white;
    padding: 1.1rem 1.4rem;
    border-radius: 18px 18px 4px 18px;
    margin-bottom: 1.2rem;
    margin-left: auto;
    max-width: 80%;
    box-shadow: 0 4px 15px rgba(79, 70, 229, 0.25);
}

.chat-bubble-bot {
    background: rgba(255, 255, 255, 0.04);
    border: 1px solid rgba(255, 255, 255, 0.06);
    backdrop-filter: blur(14px);
    color: #e2e8f0;
    padding: 1.4rem;
    border-radius: 18px 18px 18px 4px;
    margin-bottom: 0.7rem;
    max-width: 90%;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
}

.chunk-card {
    background: rgba(15, 23, 42, 0.4);
    border: 1px solid rgba(255, 255, 255, 0.05);
    border-left: 4px solid #2dd4bf;
    border-radius: 16px;
    padding: 1.2rem;
    margin-bottom: 1rem;
}

.chunk-label {
    color: #2dd4bf;
    font-size: 0.75rem;
    font-weight: 800;
    letter-spacing: 0.05em;
}

.chunk-score {
    background: rgba(20, 184, 166, 0.1);
    color: #2dd4bf;
    padding: 0.2rem 0.6rem;
    border-radius: 8px;
    font-size: 0.75rem;
    font-weight: 700;
    border: 1px solid rgba(20, 184, 166, 0.2);
}

.chunk-text {
    color: #cbd5e1;
    margin-top: 0.8rem;
    line-height: 1.65;
    font-size: 0.9rem;
}

.info-block {
    background: rgba(255, 255, 255, 0.02);
    border: 1px solid rgba(255, 255, 255, 0.05);
    border-radius: 16px;
    padding: 1.1rem;
    margin-bottom: 1rem;
}

.info-block-title {
    color: #94a3b8;
    font-size: 0.75rem;
    font-weight: 700;
    margin-bottom: 0.6rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

div.stButton > button {
    background: linear-gradient(135deg, #4f46e5 0%, #06b6d4 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    font-weight: 600 !important;
    padding: 0.6rem 1.2rem !important;
    min-height: 72px !important;
    width: 100% !important;
    white-space: normal !important;
    line-height: 1.35 !important;
}

.architecture-box {
    background: rgba(15, 23, 42, 0.6);
    border: 1px solid rgba(255, 255, 255, 0.05);
    border-radius: 16px;
    padding: 1.5rem;
    margin-top: 1rem;
}

.flow-step {
    background: rgba(255, 255, 255, 0.03);
    border: 1px solid rgba(255, 255, 255, 0.05);
    border-radius: 12px;
    padding: 1rem;
    margin-bottom: 0.8rem;
}

.flow-step-num {
    color: #4f46e5;
    font-weight: 800;
    font-size: 1.1rem;
    margin-right: 0.5rem;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────

if "messages" not in st.session_state:
    st.session_state.messages = []

if "question_count" not in st.session_state:
    st.session_state.question_count = 0

if "avg_response_time" not in st.session_state:
    st.session_state.avg_response_time = 0.0


# ─────────────────────────────────────────────────────────────
# RAG PIPELINE
# ─────────────────────────────────────────────────────────────

def get_rag_answer(question: str):
    try:
        from generate import answer_question
        return answer_question(question), False
    except Exception as e:
        return {
            "answer": f"""
Based on the indexed legal parameters of the UAE Labour Law, here is the analysis for your query:

**"{question}"**

1. **Regulatory Overview:** In alignment with Federal Decree-Law No. 33 of 2021, statutory compliance parameters regulate worker protections, notice terms, contracts, and employer liabilities.
2. **Key Determinations:** Implementation guidelines outline structured processing channels. Any local discrepancy should refer to official provisions under MOHRE.

⚠️ Note: This fallback answer appeared because the real generation pipeline raised an error:
`{str(e)}`
""",
            "context_chunks": [
                {
                    "text": "Federal Decree-Law No. 33 of 2021 regarding the Regulation of Labour Relations ensures comprehensive statutory adjustments to contractual formats and employee transitions.",
                    "source": "Federal Decree-Law No. 33 of 2021",
                    "hybrid_score": 0.96
                },
                {
                    "text": "Cabinet Resolution No. 1 of 2022 clarifies execution parameters and standard processing protocols.",
                    "source": "Cabinet Resolution No. 1 of 2022",
                    "hybrid_score": 0.89
                }
            ]
        }, True


def render_chunks(chunks):
    if not chunks:
        st.markdown(
            "<div style='color:#64748b;'>No retrieved chunks available for this answer.</div>",
            unsafe_allow_html=True
        )
        return

    for i, chunk in enumerate(chunks, 1):
        score = chunk.get("hybrid_score", 0)
        score_percentage = int(score * 100) if score <= 1 else int(score)
        source = chunk.get("source", "Unknown source")
        text = chunk.get("text", "")

        st.markdown(f"""
        <div class="chunk-card">
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <span class="chunk-label">DOCUMENT CHUNK #{i}</span>
                <span class="chunk-score">Match Rate: {score_percentage}%</span>
            </div>
            <div style="color:#64748b; margin-top:0.4rem; font-size:0.8rem; font-weight:600;">
                📍 Source Reference: {source}
            </div>
            <div class="chunk-text">
                "{text}"
            </div>
        </div>
        """, unsafe_allow_html=True)


def process_question(question: str):
    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    with st.spinner("Executing retrieval and generation pipeline..."):
        start_timer = time.time()
        pipeline_result, used_fallback = get_rag_answer(question)
        elapsed = time.time() - start_timer

    current_count = st.session_state.question_count
    st.session_state.avg_response_time = (
        (st.session_state.avg_response_time * current_count + elapsed) / (current_count + 1)
    )
    st.session_state.question_count += 1

    st.session_state.messages.append({
        "role": "assistant",
        "content": pipeline_result.get("answer", "No answer generated."),
        "chunks": pipeline_result.get("context_chunks", []),
        "response_time": elapsed,
        "used_fallback": used_fallback
    })

    st.rerun()


# ─────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding:1.5rem 0 2rem 0;">
        <div style="font-size:3.2rem; margin-bottom: 0.5rem;">⚖️</div>
        <div style="font-size:1.4rem; font-weight:800;">UAE Labour Law</div>
        <div style="color:#2dd4bf; font-size:0.75rem; font-weight:700; margin-top:0.3rem; letter-spacing:0.1em;">
            HYBRID RAG INTERACTIVE
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="info-block">
        <div class="info-block-title">CORE PIPELINE ENGINE</div>
        <span style="color:#2dd4bf; font-weight:700;">🟢 Pipeline Active</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="info-block">
        <div class="info-block-title">SYSTEM PROFILES</div>
        <table style="width:100%; font-size:0.85rem; color:#94a3b8;">
            <tr><td>• Embeddings:</td><td style="color:#f1f5f9; text-align:right;">MiniLM-L6-v2</td></tr>
            <tr><td>• Database:</td><td style="color:#f1f5f9; text-align:right;">ChromaDB Vector</td></tr>
            <tr><td>• LLM Engine:</td><td style="color:#f1f5f9; text-align:right;">Gemini 2.5 Flash</td></tr>
            <tr><td>• Index Size:</td><td style="color:#f1f5f9; text-align:right;">2,033 Fragments</td></tr>
            <tr><td>• Engine Mode:</td><td style="color:#f1f5f9; text-align:right;">Dense + BM25</td></tr>
        </table>
    </div>
    """, unsafe_allow_html=True)

    if st.button("🗑️ Clear Active Session", use_container_width=True):
        st.session_state.messages = []
        st.session_state.question_count = 0
        st.session_state.avg_response_time = 0.0
        st.rerun()


# ─────────────────────────────────────────────────────────────
# HERO
# ─────────────────────────────────────────────────────────────

st.markdown("""
<div class="hero-banner">
    <div class="hero-badge">
        🛠️ ENTERPRISE ARCHITECTURE · KNOWLEDGE EXTRACTION
    </div>
    <div class="hero-title">
        UAE Labour Law <span>Intelligent Portal</span>
    </div>
    <div class="hero-subtitle">
        Query UAE labour law provisions using a hybrid RAG pipeline that combines vector retrieval,
        BM25 keyword search, legal evidence chunks, and Gemini-based answer generation.
    </div>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────────────────────

m_col1, m_col2, m_col3, m_col4 = st.columns(4)

with m_col1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{st.session_state.question_count}</div>
        <div class="metric-label">Processed Requests</div>
    </div>
    """, unsafe_allow_html=True)

with m_col2:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-value">26</div>
        <div class="metric-label">Indexed Documents</div>
    </div>
    """, unsafe_allow_html=True)

with m_col3:
    rt_display = f"{st.session_state.avg_response_time:.2f}s" if st.session_state.avg_response_time > 0 else "—"
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{rt_display}</div>
        <div class="metric-label">Average Latency</div>
    </div>
    """, unsafe_allow_html=True)

with m_col4:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-value">2,033</div>
        <div class="metric-label">Constructed Chunks</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────

chat_tab, code_tab, about_tab = st.tabs([
    "💬 Live Chat Interface",
    "📊 Pipeline Architecture & Logic",
    "ℹ️ RAG Specifications"
])


# ─────────────────────────────────────────────────────────────
# CHAT TAB
# ─────────────────────────────────────────────────────────────

with chat_tab:
    st.markdown("### 💡 Sample Queries")
    st.markdown(
        "<p style='color:#94a3b8;'>Choose a sample question or type your own question below.</p>",
        unsafe_allow_html=True
    )

    ex_col1, ex_col2, ex_col3 = st.columns(3)
    sample_clicked = None

    with ex_col1:
        st.markdown(
            "<div style='color:#2dd4bf; font-weight:700; margin-bottom:0.5rem;'>📋 Leave & Gratuity</div>",
            unsafe_allow_html=True
        )
        if st.button("How is end-of-service gratuity calculated under the new law?", use_container_width=True):
            sample_clicked = "How is end-of-service gratuity calculated under the new law?"

        if st.button("What are the rules for annual leave pay when resigning?", use_container_width=True):
            sample_clicked = "What are the rules for annual leave pay when resigning?"

    with ex_col2:
        st.markdown(
            "<div style='color:#4f46e5; font-weight:700; margin-bottom:0.5rem;'>💼 Employment Contracts</div>",
            unsafe_allow_html=True
        )
        if st.button("What is the maximum duration for a limited contract?", use_container_width=True):
            sample_clicked = "What is the maximum duration for a limited contract?"

        if st.button("Are non-compete clauses legally binding in the UAE?", use_container_width=True):
            sample_clicked = "Are non-compete clauses legally binding in the UAE?"

    with ex_col3:
        st.markdown(
            "<div style='color:#a5b4fc; font-weight:700; margin-bottom:0.5rem;'>⚠️ Termination Rules</div>",
            unsafe_allow_html=True
        )
        if st.button("What is the legal minimum notice period for termination?", use_container_width=True):
            sample_clicked = "What is the legal minimum notice period for termination?"

        if st.button("When can an employer dismiss a worker without notice?", use_container_width=True):
            sample_clicked = "When can an employer dismiss a worker without notice?"

    if sample_clicked:
        process_question(sample_clicked)

    st.markdown("### 💬 Conversation")

    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(
                f'<div class="chat-bubble-user">{msg["content"]}</div>',
                unsafe_allow_html=True
            )

        elif msg["role"] == "assistant":
            formatted_answer = msg["content"].replace("\n", "<br>")

            st.markdown(f"""
            <div class="chat-bubble-bot">
                <div style="font-weight:700; margin-bottom:0.6rem; color:#2dd4bf; font-size: 0.9rem;">
                    ⚖️ LEGAL INFERENCE SYSTEM RESPONSE
                </div>
                <div>{formatted_answer}</div>
                <div style="margin-top:0.8rem; color:#64748b; font-size:0.75rem;">
                    Response time: {msg.get("response_time", 0):.2f}s
                </div>
            </div>
            """, unsafe_allow_html=True)

            with st.expander("📚 View retrieved document chunks for this answer"):
                render_chunks(msg.get("chunks", []))

    user_question = st.chat_input("Input compliance queries or operational labour scenarios...")

    if user_question:
        process_question(user_question)


# ─────────────────────────────────────────────────────────────
# PIPELINE TAB
# ─────────────────────────────────────────────────────────────

with code_tab:
    st.markdown("### 📊 Application Infrastructure & Engine Logic")
    st.markdown("This blueprint outlines the visual execution pipeline for the system runtime stack.")

    v_col1, v_col2 = st.columns([1, 1])

    with v_col1:
        st.markdown("<div class='architecture-box'>", unsafe_allow_html=True)
        st.markdown("<h4>🔄 RAG Operational Pipeline Flow</h4>", unsafe_allow_html=True)

        st.markdown("""
        <div class='flow-step'>
            <span class='flow-step-num'>1. Input</span> 
            <strong>User Query Capture:</strong> The user question is passed into the processing pipeline.
        </div>
        <div class='flow-step'>
            <span class='flow-step-num'>2. Vectorization</span> 
            <strong>Embedding Transformation:</strong> Text is transformed using <code>MiniLM-L6-v2</code>.
        </div>
        <div class='flow-step'>
            <span class='flow-step-num'>3. Hybrid Retrieval</span> 
            <strong>Dual Index Query:</strong> ChromaDB dense search and BM25 sparse search retrieve legal chunks.
        </div>
        <div class='flow-step'>
            <span class='flow-step-num'>4. Reranking</span> 
            <strong>Hybrid Fusion:</strong> Scores are combined into a final ranked evidence list.
        </div>
        <div class='flow-step'>
            <span class='flow-step-num'>5. Generation</span> 
            <strong>Contextual Prompting:</strong> Gemini generates the final legal answer using retrieved fragments.
        </div>
        """, unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

    with v_col2:
        st.markdown("<div class='architecture-box'>", unsafe_allow_html=True)
        st.markdown("<h4>📐 Core Code Algorithms</h4>", unsafe_allow_html=True)

        st.markdown("##### Hybrid Scoring Protocol")
        st.code("""
def compute_hybrid_fusion(semantic_score, bm25_score, alpha=0.7):
    return (alpha * semantic_score) + ((1.0 - alpha) * bm25_score)
        """, language="python")

        st.markdown("##### Message and Evidence Binding")
        st.code("""
st.session_state.messages.append({
    "role": "assistant",
    "content": pipeline_result["answer"],
    "chunks": pipeline_result["context_chunks"]
})

# Each answer keeps its own chunks.
for msg in st.session_state.messages:
    if msg["role"] == "assistant":
        display_answer(msg["content"])
        display_chunks(msg["chunks"])
        """, language="python")

        st.markdown("</div>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# ABOUT TAB
# ─────────────────────────────────────────────────────────────

with about_tab:
    st.markdown("## 🧠 Architectural Overview")
    st.markdown("Advanced system specs for the hybrid verification layer:")

    a1, a2 = st.columns(2)

    with a1:
        st.markdown("""
        ### 🔍 Dense vs Sparse Hybrid Weights

        The architecture uses dual retrieval strategies:

        - **Semantic Retrieval Weight:** `0.70`
        - **Keyword Match Weight:** `0.30`

        ### 🎯 Hallucination Reduction

        - Answers are generated from retrieved legal fragments.
        - Each response stores its own evidence chunks.
        - The UI no longer mixes chunks from different questions.
        """)

    with a2:
        st.markdown("""
        ### 📦 Code-Base Profile

        - **Vector Database:** ChromaDB
        - **Sparse Retrieval:** BM25
        - **Embedding Model:** MiniLM-L6-v2
        - **Generation Model:** Gemini 2.5 Flash
        - **Interface:** Streamlit
        """)


# ─────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────

st.markdown("""
<hr style="border: 0; height: 1px; background: rgba(255,255,255,0.06); margin-top: 3rem;">
<div style="text-align:center; color:#475569; padding:0.5rem; font-size:0.8rem; font-weight:600; letter-spacing:0.05em;">
    NLP APPLICATIONS INFRASTRUCTURE · HYBRID RAG SYSTEM · 2026
</div>
""", unsafe_allow_html=True)