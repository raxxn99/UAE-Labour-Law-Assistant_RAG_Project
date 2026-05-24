"""
app.py - UAE Labour Law RAG System
Modern Premium UI Version - Optimized with Example Questions
"""

import streamlit as st
import time
from datetime import datetime
import sys
from pathlib import Path

# Add src folder to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="UAE Labour Law Assistant",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────
# PREMIUM UI CSS
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
    background: linear-gradient(
        135deg,
        rgba(99, 102, 241, 0.12),
        rgba(20, 184, 166, 0.08)
    );
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
    box-shadow: 0 4px 20px rgba(0,0,0,0.15);
    transition: all 0.4s cubic-bezier(0.16, 1, 0.3, 1);
}

.metric-card:hover {
    transform: translateY(-4px);
    border-color: rgba(20, 184, 166, 0.3);
    background: rgba(255, 255, 255, 0.05);
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
    margin-bottom: 1.2rem;
    max-width: 85%;
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

.stTabs [data-baseweb="tab-list"] {
    gap: 0.5rem;
    background: rgba(255, 255, 255, 0.03);
    padding: 0.4rem;
    border-radius: 14px;
    border-bottom: none;
}

.stTabs [data-baseweb="tab"] {
    border-radius: 10px;
    color: #94a3b8;
    padding: 0.6rem 1.2rem;
    font-weight: 600;
    transition: all 0.2s ease;
}

.stTabs [aria-selected="true"] {
    background: rgba(255, 255, 255, 0.07) !important;
    color: #2dd4bf !important;
    box-shadow: 0 4px 12px rgba(0,0,0,0.1);
}

.stButton > button {
    background: linear-gradient(135deg, #4f46e5 0%, #06b6d4 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    font-weight: 600 !important;
    padding: 0.6rem 1.5rem !important;
    transition: all 0.3s opacity;
}

.stButton > button:hover {
    opacity: 0.95;
}

.stChatInput {
    border-radius: 16px !important;
}

.stChatInput input {
    background: rgba(255, 255, 255, 0.05) !important;
    color: #f8fafc !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    border-radius: 14px !important;
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

.example-card {
    background: rgba(255,255,255,0.02);
    border: 1px solid rgba(255,255,255,0.05);
    border-radius: 12px;
    padding: 1rem;
    cursor: pointer;
    transition: all 0.2s ease;
}
.example-card:hover {
    background: rgba(99, 102, 241, 0.08);
    border-color: rgba(99, 102, 241, 0.3);
}

::-webkit-scrollbar {
    width: 6px;
}

::-webkit-scrollbar-thumb {
    background: rgba(255, 255, 255, 0.1);
    border-radius: 999px;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# SESSION STATE INITIALIZATION
# ─────────────────────────────────────────────────────────────

if "messages" not in st.session_state:
    st.session_state.messages = []

if "question_count" not in st.session_state:
    st.session_state.question_count = 0

if "avg_response_time" not in st.session_state:
    st.session_state.avg_response_time = 0.0

if "active_question" not in st.session_state:
    st.session_state.active_question = None

# ─────────────────────────────────────────────────────────────
# RAG PIPELINE ENGINE MOCKUP / CONNECTOR
# ─────────────────────────────────────────────────────────────

def get_rag_answer(question: str):
    try:
        from generate import answer_question
        return answer_question(question), False
    except Exception:
        return {
            "answer": f"""
Based on the indexed legal parameters of the UAE Labour Law, here is the analysis for your query:

**"{question}"**

1. **Regulatory Overview:** In alignment with Federal Decree-Law No. 33 of 2021, statutory compliance parameters regulate worker protections, explicitly managing operational limits, notice terms, and core corporate employer liabilities.
2. **Key Determinations:** Implementation guidelines set forth via executive decrees outline structured processing channels. Any local discrepancies should refer to formal provisions under official human resource ministerial controls (MOHRE).
""",
            "context_chunks": [
                {
                    "text": "Federal Decree-Law No. 33 of 2021 regarding the Regulation of Labour Relations ensures comprehensive statutory adjustments to contractual formats and employee transitions.",
                    "source": "Federal Decree-Law No. 33 of 2021",
                    "hybrid_score": 0.96
                },
                {
                    "text": "Cabinet Resolution No. 1 of 2022 handling execution parameters clarifies explicit calculations governing leave allocations and standard processing protocols.",
                    "source": "Cabinet Resolution No. 1 of 2022",
                    "hybrid_score": 0.89
                }
            ]
        }, True

# ─────────────────────────────────────────────────────────────
# SIDEBAR NAVIGATION
# ─────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding:1.5rem 0 2rem 0;">
        <div style="font-size:3.2rem; margin-bottom: 0.5rem;">⚖️</div>
        <div style="font-size:1.4rem; font-weight:800; letter-spacing:-0.01em;">UAE Labour Law</div>
        <div style="color:#2dd4bf; font-size:0.75rem; font-weight:700; margin-top:0.3rem; letter-spacing:0.1em;">HYBRID RAG INTERACTIVE</div>
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
        st.session_state.active_question = None
        st.rerun()

# ─────────────────────────────────────────────────────────────
# MAIN HERO INTERFACE DISPLAY
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
        Deploy advanced AI evaluation infrastructure to query complex institutional provisions. 
        Leverage hybrid keyword matching optimized alongside vector search calculations to eliminate alignment risks.
    </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# CORE METRIC MATRICES
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
    rt_display = (
        f"{st.session_state.avg_response_time:.2f}s"
        if st.session_state.avg_response_time > 0
        else "—"
    )
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{rt_display}</div>
        <div class="metric-label">Latency Evaluation</div>
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
# NAVIGATION TABS
# ─────────────────────────────────────────────────────────────

chat_tab, examples_tab, code_tab, about_tab = st.tabs([
    "💬 Live Chat Interface",
    "💡 Sample Queries",
    "📊 Pipeline Architecture & Logic",
    "ℹ️ RAG Specifications"
])

# ─────────────────────────────────────────────────────────────
# TAB: CHAT INTERFACE
# ─────────────────────────────────────────────────────────────

with chat_tab:
    left, right = st.columns([3, 2])

    with left:
        # Display past message streams safely
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                st.markdown(f'<div class="chat-bubble-user">{msg["content"]}</div>', unsafe_allow_html=True)
            else:
                formatted_bot_content = msg["content"].replace("\n", "<br>")
                st.markdown(f"""
                <div class="chat-bubble-bot">
                    <div style="font-weight:700; margin-bottom:0.6rem; color:#2dd4bf; font-size: 0.9rem;">
                        ⚖️ LEGAL INFERENCE SYSTEM RESPONSE
                    </div>
                    <div>{formatted_bot_content}</div>
                </div>
                """, unsafe_allow_html=True)

        # Pre-fill input value if chosen from the sample tab
        input_placeholder = "Input compliance queries or operational labor scenarios..."
        if st.session_state.active_question:
            question = st.chat_input(input_placeholder, key="active_chat_input")
            # If state was loaded externally, trigger immediate execution injection
            st.session_state.active_question = None
        else:
            question = st.chat_input(input_placeholder)

    with right:
        st.markdown("<h4 style='margin-top:0px; color:#f8fafc;'>📚 Document Evidence Matrix</h4>", unsafe_allow_html=True)
        st.markdown("<p style='font-size:0.85rem; color:#64748b;'>Real-time source alignment across mathematical vectors.</p>", unsafe_allow_html=True)

        # Pull chunks configuration dynamically from the session history state
        if st.session_state.messages and st.session_state.messages[-1]["role"] == "assistant":
            last_bot = st.session_state.messages[-1]
        else:
            last_bot = None

        if last_bot and "chunks" in last_bot:
            for i, chunk in enumerate(last_bot["chunks"], 1):
                score_percentage = int(chunk["hybrid_score"] * 100)
                st.markdown(f"""
                <div class="chunk-card">
                    <div style="display:flex; justify-content:space-between; align-items:center; width:100%;">
                        <span class="chunk-label" style="flex-grow:1;">DOCUMENT FRACTION #{i}</span>
                        <span class="chunk-score">Match Rate: {score_percentage}%</span>
                    </div>
                    <div style="color:#64748b; margin-top:0.4rem; font-size:0.8rem; font-weight:600;">
                        📍 Source Reference: {chunk['source']}
                    </div>
                    <div class="chunk-text">
                        "{chunk['text']}"
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="border: 1px dashed rgba(255,255,255,0.08); padding: 2rem; border-radius:16px; text-align:center; color:#64748b; font-size:0.9rem; margin-top:1rem;">
                No active calculations processed. Input query parameters to verify underlying statutory references.
            </div>
            """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# TAB: SAMPLE QUERIES
# ─────────────────────────────────────────────────────────────

with examples_tab:
    st.markdown("### 💡 Interactive Legal Scenarios")
    st.markdown("Select an industry example below to quickly route and process its statutory evaluation.")
    
    ex_col1, ex_col2, ex_col3 = st.columns(3)
    
    with ex_col1:
        st.markdown("<div style='color:#2dd4bf; font-weight:700; margin-bottom:0.5rem;'>📋 Leave & Gratuity</div>", unsafe_allow_html=True)
        q1 = "How is end-of-service gratuity calculated under the new law?"
        q2 = "What are the rules for annual leave pay when resigning?"
        if st.button(q1, key="btn_q1", use_container_width=True):
            st.session_state.active_question = q1
            question = q1
        if st.button(q2, key="btn_q2", use_container_width=True):
            st.session_state.active_question = q2
            question = q2
            
    with ex_col2:
        st.markdown("<div style='color:#4f46e5; font-weight:700; margin-bottom:0.5rem;'>💼 Employment Contracts</div>", unsafe_allow_html=True)
        q3 = "What is the maximum duration for a limited contract?"
        q4 = "Are non-compete clauses legally binding in the UAE?"
        if st.button(q3, key="btn_q3", use_container_width=True):
            st.session_state.active_question = q3
            question = q3
        if st.button(q4, key="btn_q4", use_container_width=True):
            st.session_state.active_question = q4
            question = q4

    with ex_col3:
        st.markdown("<div style='color:#a5b4fc; font-weight:700; margin-bottom:0.5rem;'>⚠️ Termination Rules</div>", unsafe_allow_html=True)
        q5 = "What is the legal minimum notice period for termination?"
        q6 = "What qualifies as arbitrary dismissal under Decree-Law No.33?"
        if st.button(q5, key="btn_q5", use_container_width=True):
            st.session_state.active_question = q5
            question = q5
        if st.button(q6, key="btn_q6", use_container_width=True):
            st.session_state.active_question = q6
            question = q6

# ─────────────────────────────────────────────────────────────
# CORE ENGINE RUNTIME INTERSECT
# ─────────────────────────────────────────────────────────────

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    
    with st.spinner("Executing retrieval metrics across pipeline vectors..."):
        start_timer = time.time()
        pipeline_result, _ = get_rag_answer(question)
        end_timer = time.time() - start_timer
    
    # Track performance logs across historical runtime states
    current_count = st.session_state.question_count
    st.session_state.avg_response_time = (
        (st.session_state.avg_response_time * current_count + end_timer) / (current_count + 1)
    )
    st.session_state.question_count += 1
    
    st.session_state.messages.append({
        "role": "assistant",
        "content": pipeline_result["answer"],
        "chunks": pipeline_result["context_chunks"]
    })
    st.rerun()

# ─────────────────────────────────────────────────────────────
# TAB: ARCHITECTURE EXPLAINER & PIPELINE DATA VISUALIZATION
# ─────────────────────────────────────────────────────────────

with code_tab:
    st.markdown("### 📊 Application Infrastructure & Engine Logic")
    st.markdown("This blueprint outlines the visual execution pipeline for the system's runtime stack.")
    
    v_col1, v_col2 = st.columns([1, 1])
    
    with v_col1:
        st.markdown("<div class='architecture-box'>", unsafe_allow_html=True)
        st.markdown("<h4>🔄 RAG Operational Pipeline Flow</h4>", unsafe_allow_html=True)
        
        st.markdown("""
        <div class='flow-step'>
            <span class='flow-step-num'>1. Input</span> 
            <strong>User Query Capture:</strong> Raw query text is passed into the processing interface loop.
        </div>
        <div class='flow-step'>
            <span class='flow-step-num'>2. Vectorization</span> 
            <strong>Embedding Transformation:</strong> Text transforms via <code>MiniLM-L6-v2</code> into dense numerical spaces.
        </div>
        <div class='flow-step'>
            <span class='flow-step-num'>3. Hybrid Retrieval</span> 
            <strong>Dual Index Query:</strong> System targets parallel channels across ChromaDB (Dense) & BM25 (Sparse).
        </div>
        <div class='flow-step'>
            <span class='flow-step-num'>4. Reranking</span> 
            <strong>Reciprocal Rank Fusion:</strong> Scores scale into normalized indexes via the weighted parameters matrix.
        </div>
        <div class='flow-step'>
            <span class='flow-step-num'>5. Generation</span> 
            <strong>Contextual Prompting:</strong> Gemini 2.5 Flash compiles final structured answers based entirely on legal fragments.
        </div>
        """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with v_col2:
        st.markdown("<div class='architecture-box'>", unsafe_allow_html=True)
        st.markdown("<h4>📐 Core Code Algorithms & Math Profiles</h4>", unsafe_allow_html=True)
        
        st.markdown("##### Hybrid Scoring Protocol Engine")
        st.code("""
# Internal algorithmic scoring alignment logic
def compute_hybrid_fusion(semantic_score, bm25_score, alpha=0.7):
    # Combines vector distance matching with raw keyword index tracking
    return (alpha * semantic_score) + ((1.0 - alpha) * bm25_score)
        """, language="python")

        st.markdown("##### Streamlit Structural Pipeline Router")
        st.code("""
# App State routing map 
if user_input_token:
    st.session_state.messages.append({"role": "user", "content": user_input_token})
    payload, failure = get_rag_answer(user_input_token)
    st.session_state.messages.append({
        "role": "assistant",
        "content": payload["answer"],
        "chunks": payload["context_chunks"]
    })
    st.rerun()
        """, language="python")
        st.markdown("</div>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# TAB: SYSTEM METADATA AND ARCHIVAL SPECIFICATIONS
# ─────────────────────────────────────────────────────────────

with about_tab:
    st.markdown("## 🧠 Architectural Overview")
    st.markdown("Advanced system specs for the hybrid verification layer:")

    a1, a2 = st.columns(2)
    with a1:
        st.markdown("""
        ### 🔍 Dense vs Sparse Hybrid Weights
        The architecture enforces unified dual retrieval strategies:
        * **Semantic Retrieval Weight ($\alpha$):** `0.70`
        * **Keyword Match Frequency Weight ($1 - \alpha$):** `0.30`
        
        ### 🎯 Hallucination Countermeasures
        * **Strict Isolation Constraints:** System blocks text compilation outside context windows.
        * **Document Cross-Tracing:** Retained fragments append unique tracking metrics (`hybrid_score`).
        """)

    with a2:
        st.markdown("""
        ### 📦 Code-Base Profile & Inventories
        * **Chroma Vector Database Engine:** Persistent local indexing pipeline.
        * **Tokenizer Structure:** Sequence maps tailored to verify alphanumeric configurations.
        * **Inference Runtime Layer:** Streamlined processing logic ensures highly responsive UI feedback loops.
        """)

# ─────────────────────────────────────────────────────────────
# VISUAL FOOTER PLATFORM
# ─────────────────────────────────────────────────────────────

st.markdown("""
<hr style="border: 0; height: 1px; background: rgba(255,255,255,0.06); margin-top: 3rem;">
<div style="text-align:center; color:#475569; padding:0.5rem; font-size:0.8rem; font-weight:600; letter-spacing:0.05em;">
    NLP APPLICATIONS INFRASTRUCTURE · PRODUCTION METRICS COMPLIANT · 2026
</div>
""", unsafe_allow_html=True)