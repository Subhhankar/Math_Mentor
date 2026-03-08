import streamlit as st
import json
import time
import os
from pathlib import Path
from hitl import (
    should_trigger_hitl,
    create_review,
    approve_review,
    edit_review,
    reject_review,
    save_student_feedback,
    get_pending_reviews,
)
from memory import (
    save_memory,
    retrieve_similar,
    format_memory_context,
    apply_ocr_asr_corrections,
    save_ocr_asr_correction,
    update_feedback,
    get_stats,
)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Math Mentor",
    page_icon="∑",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Mono:wght@400;500&family=DM+Sans:wght@300;400;500;600&display=swap');

:root {
    --bg:        #0d0f14;
    --surface:   #151820;
    --border:    #252a35;
    --accent:    #6ee7b7;
    --accent2:   #818cf8;
    --warn:      #fbbf24;
    --danger:    #f87171;
    --text:      #e2e8f0;
    --muted:     #64748b;
    --pass:      #34d399;
    --fail:      #f87171;
}

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: var(--bg);
    color: var(--text);
}

/* ── Header ── */
.mentor-header {
    display: flex;
    align-items: baseline;
    gap: 12px;
    padding: 2rem 0 1.5rem;
    border-bottom: 1px solid var(--border);
    margin-bottom: 2rem;
}
.mentor-title {
    font-family: 'DM Serif Display', serif;
    font-size: 2.4rem;
    color: var(--accent);
    letter-spacing: -0.5px;
    margin: 0;
}
.mentor-sub {
    font-size: 0.85rem;
    color: var(--muted);
    font-family: 'DM Mono', monospace;
}

/* ── Mode selector ── */
.mode-strip {
    display: flex;
    gap: 8px;
    margin-bottom: 1.5rem;
}
.mode-btn {
    flex: 1;
    padding: 10px;
    border: 1px solid var(--border);
    border-radius: 8px;
    background: var(--surface);
    color: var(--muted);
    font-family: 'DM Mono', monospace;
    font-size: 0.8rem;
    cursor: pointer;
    text-align: center;
    transition: all 0.2s;
}
.mode-btn.active {
    border-color: var(--accent);
    color: var(--accent);
    background: rgba(110,231,183,0.06);
}

/* ── Cards ── */
.card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.2rem 1.4rem;
    margin-bottom: 1rem;
}
.card-title {
    font-family: 'DM Mono', monospace;
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    color: var(--muted);
    margin-bottom: 0.8rem;
}
.card-title span {
    color: var(--accent);
}

/* ── Agent trace ── */
.trace-row {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 6px 0;
    border-bottom: 1px solid var(--border);
    font-size: 0.85rem;
}
.trace-row:last-child { border-bottom: none; }
.trace-dot {
    width: 8px; height: 8px;
    border-radius: 50%;
    flex-shrink: 0;
}
.dot-pass  { background: var(--pass); }
.dot-fail  { background: var(--fail); }
.dot-warn  { background: var(--warn); }
.dot-run   { background: var(--accent2); }
.trace-name { color: var(--text); font-family: 'DM Mono', monospace; min-width: 160px; }
.trace-note { color: var(--muted); font-size: 0.78rem; }

/* ── Confidence bar ── */
.conf-wrap { margin: 6px 0 4px; }
.conf-label {
    display: flex;
    justify-content: space-between;
    font-size: 0.78rem;
    color: var(--muted);
    margin-bottom: 4px;
}
.conf-bar-bg {
    background: var(--border);
    border-radius: 99px;
    height: 6px;
    width: 100%;
}
.conf-bar-fill {
    height: 6px;
    border-radius: 99px;
    transition: width 0.5s ease;
}

/* ── Steps ── */
.step-block {
    border-left: 2px solid var(--accent2);
    padding: 8px 0 8px 14px;
    margin-bottom: 10px;
}
.step-num {
    font-family: 'DM Mono', monospace;
    font-size: 0.68rem;
    color: var(--accent2);
    text-transform: uppercase;
    letter-spacing: 1px;
}
.step-heading {
    font-weight: 600;
    font-size: 0.9rem;
    margin: 2px 0;
}
.step-exp { font-size: 0.85rem; color: var(--muted); margin-bottom: 4px; }
.step-math {
    font-family: 'DM Mono', monospace;
    font-size: 0.82rem;
    background: rgba(129,140,248,0.08);
    padding: 6px 10px;
    border-radius: 6px;
    margin: 4px 0;
}
.step-tip {
    font-size: 0.78rem;
    color: var(--warn);
    margin-top: 4px;
}

/* ── Answer box ── */
.answer-box {
    background: rgba(110,231,183,0.07);
    border: 1px solid rgba(110,231,183,0.25);
    border-radius: 10px;
    padding: 1rem 1.2rem;
    margin: 0.6rem 0;
}
.answer-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    color: var(--accent);
    margin-bottom: 6px;
}
.answer-value {
    font-family: 'DM Serif Display', serif;
    font-size: 1.6rem;
    color: var(--accent);
}

/* ── Context chunks ── */
.chunk {
    font-family: 'DM Mono', monospace;
    font-size: 0.76rem;
    color: var(--muted);
    background: rgba(255,255,255,0.03);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 8px 10px;
    margin-bottom: 6px;
    line-height: 1.5;
}

/* ── Feedback ── */
.fb-row {
    display: flex;
    gap: 10px;
    margin-top: 0.8rem;
}
.fb-btn {
    flex: 1;
    padding: 10px;
    border-radius: 8px;
    border: 1px solid var(--border);
    font-size: 0.85rem;
    font-family: 'DM Sans', sans-serif;
    cursor: pointer;
    text-align: center;
    transition: all 0.18s;
}
.fb-correct {
    border-color: var(--pass);
    color: var(--pass);
    background: rgba(52,211,153,0.06);
}
.fb-incorrect {
    border-color: var(--danger);
    color: var(--danger);
    background: rgba(248,113,113,0.06);
}

/* ── HITL banner ── */
.hitl-banner {
    background: rgba(251,191,36,0.08);
    border: 1px solid rgba(251,191,36,0.35);
    border-radius: 10px;
    padding: 0.9rem 1.1rem;
    margin: 0.8rem 0;
    font-size: 0.85rem;
    color: var(--warn);
}
.hitl-banner b { font-weight: 600; }

/* ── Streamlit overrides ── */
.stTextArea textarea, .stTextInput input {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
    border-radius: 8px !important;
    font-family: 'DM Sans', sans-serif !important;
}
.stButton > button {
    background: var(--accent) !important;
    color: #0d0f14 !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.82rem !important;
    font-weight: 500 !important;
    padding: 0.55rem 1.4rem !important;
    letter-spacing: 0.5px;
}
.stButton > button:hover { opacity: 0.88 !important; }
.stFileUploader {
    background: var(--surface) !important;
    border: 1px dashed var(--border) !important;
    border-radius: 10px !important;
}
div[data-testid="stExpander"] {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
}
.stSelectbox > div > div {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
}
.stTabs [data-baseweb="tab-list"] {
    background: var(--surface) !important;
    border-radius: 8px;
    gap: 4px;
    padding: 4px;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'DM Mono', monospace !important;
    font-size: 0.78rem !important;
    color: var(--muted) !important;
    border-radius: 6px !important;
}
.stTabs [aria-selected="true"] {
    background: rgba(110,231,183,0.1) !important;
    color: var(--accent) !important;
}
section[data-testid="stSidebar"] { display: none !important; }
#MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ── Session state ─────────────────────────────────────────────────────────────
defaults = {
    "input_mode": "Text",
    "raw_input": "",
    "extracted_text": "",
    "parsed": None,
    "routing": None,
    "solution": None,
    "verification": None,
    "explanation": None,
    "pipeline_ran": False,
    "feedback_given": None,
    "feedback_comment": "",
    "show_comment_box": False,
    # HITL
    "hitl_triggered": False,
    "hitl_reason": "",
    "hitl_record": None,
    "hitl_resolved": False,
    # Memory
    "memory_id": None,
    "similar_problems": [],
    "last_ocr_key": None,
    "want_explanation": False,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ── Pipeline runner (swap with real agents) ───────────────────────────────────
def run_pipeline(raw_text: str, input_mode: str) -> dict:
    """
    Replace each mock_* call with your real agent call.
    All agents accept/return the same dicts shown here.
    """
    from parser_agent        import ParserAgent
    from intent_router_agent import IntentRouterAgent
    from solver_agent        import SolverAgent
    from vector_store        import get_retriever
    from verifier_agent      import VerifierAgent
    from explainer_agent     import ExplainerAgent
    retriever = get_retriever(k=3)

    # Import your retriever
    # from your_rag_setup import retriever

    source_map = {"Text": "text", "Image": "ocr", "Audio": "asr"}
    source = source_map.get(input_mode, "text")

    gemini_key = os.getenv("GOOGLE_API_KEY", "")

    # Apply known OCR/ASR correction rules before parsing
    cleaned_text = apply_ocr_asr_corrections(raw_text, source)

    parsed  = ParserAgent(gemini_key).parse(cleaned_text, source)
    routing = IntentRouterAgent(gemini_key).route(parsed)

    # Retrieve similar solved problems from memory
    similar  = retrieve_similar(parsed.get("problem_text", ""), parsed.get("topic", "other"))
    mem_ctx  = format_memory_context(similar)

    solution     = SolverAgent(retriever).solve(routing, memory_context=mem_ctx)
    verification = VerifierAgent().verify(routing, solution)
    explanation  = ExplainerAgent().explain(routing, solution, verification)

    # Save to memory
    mem_record = save_memory(
        raw_input=raw_text,
        input_source=source,
        parsed=parsed,
        solution=solution,
        verification=verification,
        retrieved_docs=solution.get("retrieved_docs", []),
    )

    return {
        "parsed": parsed,
        "routing": routing,
        "solution": solution,
        "verification": verification,
        "explanation": explanation,
        "memory_id": mem_record["memory_id"],
        "similar_problems": similar,
    }


def run_mock_pipeline(raw_text: str, input_mode: str) -> dict:
    """Mock pipeline for UI testing without API keys."""
    time.sleep(1.2)
    parsed = {
        "problem_text": raw_text,
        "topic": "probability",
        "variables": ["X", "Y"],
        "constraints": ["P(X)=0.3", "P(Y)=0.5"],
        "needs_clarification": False,
        "clarification_reason": "",
        "input_source": input_mode.lower(),
    }
    routing = {
        "problem_text": raw_text,
        "topic": "probability",
        "sub_type": "union of independent events",
        "intent": "solve",
        "strategy": ["rag_retrieval", "python_calculator"],
        "requires_calculator": True,
        "requires_rag": True,
        "out_of_scope": False,
        "routing_notes": "Standard probability union rule identified.",
    }
    solution = {
        "solution": {
            "steps": [
                {"step_number": 1, "description": "Identify independence",
                 "work": "X ⊥ Y given", "reasoning": "Problem states independence"},
                {"step_number": 2, "description": "Apply union formula",
                 "work": "P(X∪Y) = P(X) + P(Y) − P(X∩Y)", "reasoning": "Inclusion-exclusion"},
                {"step_number": 3, "description": "Compute intersection",
                 "work": "P(X∩Y) = 0.3 × 0.5 = 0.15", "reasoning": "Independence → multiply"},
                {"step_number": 4, "description": "Final calculation",
                 "work": "P(X∪Y) = 0.3 + 0.5 − 0.15 = 0.65", "reasoning": "Substitute"},
            ],
            "final_answer": "0.65",
            "answer_latex": "P(X \\cup Y) = 0.65",
        },
        "explanation": "We use inclusion-exclusion for independent events.",
        "formulas_used": ["P(A∪B) = P(A)+P(B)−P(A∩B)", "P(A∩B)=P(A)·P(B) if independent"],
        "confidence": 0.93,
        "confidence_reason": "Classic textbook problem, well-supported by KB.",
        "needs_human_review": False,
        "review_reason": "",
        "edge_cases": "Assumes independence as stated.",
        "sources_used": ["Probability Formulas Sheet", "Independence Definitions"],
        "retrieved_docs": [
            "For independent events A and B, P(A∩B) = P(A)·P(B)...",
            "The addition rule states P(A∪B) = P(A)+P(B)−P(A∩B)...",
            "Union probability cannot exceed 1 and must be ≥ max(P(A), P(B))...",
        ],
        "calculator_trace": [
            {"expression": "0.3 * 0.5", "result": 0.15, "error": None},
            {"expression": "0.3 + 0.5 - 0.15", "result": 0.65, "error": None},
        ],
    }
    verification = {
        "verdict": "pass",
        "correctness": {"status": "pass", "issues": []},
        "units_and_domain": {"status": "pass", "issues": []},
        "edge_cases": {"status": "pass", "issues": []},
        "confidence": 0.95,
        "confidence_reason": "All checks passed.",
        "needs_hitl": False,
        "hitl_reason": "",
        "suggested_fix": "",
    }
    explanation = {
        "title": "Probability of Union of Two Independent Events",
        "concept_intro": "When two events are independent, knowing one happened tells you nothing about the other. We use the inclusion-exclusion principle to find the probability of either happening.",
        "steps": [
            {
                "step_number": 1,
                "heading": "Recognise independence",
                "explanation": "The problem tells us X and Y are independent — this is the key fact we'll use.",
                "math": "X ⊥ Y",
                "tip": "",
            },
            {
                "step_number": 2,
                "heading": "Write the union formula",
                "explanation": "For any two events, P(A∪B) = P(A) + P(B) − P(A∩B). We subtract the intersection to avoid counting it twice.",
                "math": "P(X∪Y) = P(X) + P(Y) − P(X∩Y)",
                "tip": "Students often forget to subtract the intersection — don't!",
            },
            {
                "step_number": 3,
                "heading": "Compute the intersection",
                "explanation": "Because X and Y are independent, we simply multiply their individual probabilities.",
                "math": "P(X∩Y) = 0.3 × 0.5 = 0.15",
                "tip": "",
            },
            {
                "step_number": 4,
                "heading": "Substitute and solve",
                "explanation": "Now plug everything into the formula.",
                "math": "P(X∪Y) = 0.3 + 0.5 − 0.15 = 0.65",
                "tip": "",
            },
        ],
        "final_answer": "0.65",
        "summary": "We identified the events as independent, applied the inclusion-exclusion principle, and computed the intersection by multiplying probabilities. The final answer is 0.65.",
        "key_concepts": ["Independence", "Addition rule", "Inclusion-exclusion"],
        "common_mistakes": ["Forgetting to subtract intersection", "Assuming P(A∩B)=0 (mutually exclusive vs independent)"],
        "difficulty": "medium",
        "topic": "probability",
        "follow_up_problems": [
            "P(A)=0.4, P(B)=0.6, independent. Find P(A∪B).",
            "Two dice rolled. Find P(first shows 3 OR second shows 5).",
        ],
        "needs_hitl": False,
    }
    return {
        "parsed": parsed,
        "routing": routing,
        "solution": solution,
        "verification": verification,
        "explanation": explanation,
    }


# ── UI helpers ────────────────────────────────────────────────────────────────
def latex_to_plain(text: str) -> str:
    """Strip LaTeX markup and convert to plain readable math."""
    import re
    if not text:
        return text
    text = re.sub(r'\$+', '', text)
    text = re.sub(r'\\frac\{([^}]+)\}\{([^}]+)\}', r'(\1)/(\2)', text)
    text = re.sub(r'\\sqrt\{([^}]+)\}', r'sqrt(\1)', text)
    text = re.sub(r'\\int_\{([^}]+)\}\^\{([^}]+)\}', r'integral from \1 to \2 of', text)
    text = re.sub(r'\\int', 'integral', text)
    text = re.sub(r'\\sum_\{([^}]+)\}\^\{([^}]+)\}', r'sum from \1 to \2 of', text)
    text = re.sub(r'\\sum', 'sum', text)
    text = re.sub(r'\\cdot', '×', text)
    text = re.sub(r'\\times', '×', text)
    text = re.sub(r'\\pm', '±', text)
    text = re.sub(r'\\leq', '≤', text)
    text = re.sub(r'\\geq', '≥', text)
    text = re.sub(r'\\neq', '≠', text)
    text = re.sub(r'\\infty', '∞', text)
    text = re.sub(r'\\pi', 'π', text)
    text = re.sub(r'\\alpha', 'α', text)
    text = re.sub(r'\\beta', 'β', text)
    text = re.sub(r'\\theta', 'θ', text)
    text = re.sub(r'\\left|\\right', '', text)
    text = re.sub(r'\\\w+', '', text)
    text = re.sub(r'[{}]', '', text)
    text = re.sub(r' +', ' ', text).strip()
    return text


def confidence_bar(score: float, label: str = "Confidence"):
    pct = int(score * 100)
    if score >= 0.85:
        color = "#34d399"
    elif score >= 0.65:
        color = "#fbbf24"
    else:
        color = "#f87171"
    st.markdown(f"""
    <div class="conf-wrap">
      <div class="conf-label"><span>{label}</span><span>{pct}%</span></div>
      <div class="conf-bar-bg">
        <div class="conf-bar-fill" style="width:{pct}%; background:{color};"></div>
      </div>
    </div>""", unsafe_allow_html=True)


def agent_trace(parsed, routing, solution, verification, explanation):
    st.markdown('<div class="card"><div class="card-title">⬡ <span>Agent Trace</span></div>', unsafe_allow_html=True)

    def row(dot_cls, name, note):
        st.markdown(f"""
        <div class="trace-row">
          <div class="trace-dot {dot_cls}"></div>
          <span class="trace-name">{name}</span>
          <span class="trace-note">{note}</span>
        </div>""", unsafe_allow_html=True)

    # Parser
    p_status = "dot-warn" if parsed.get("needs_clarification") else "dot-pass"
    p_note = parsed.get("clarification_reason") or f"topic → {parsed.get('topic')} · source → {parsed.get('input_source')}"
    row(p_status, "1 · ParserAgent", p_note)

    # Router
    r_note = f"intent → {routing.get('intent')} · strategy → {', '.join(routing.get('strategy', []))}"
    row("dot-run", "2 · IntentRouterAgent", r_note)

    # Solver
    calc = routing.get("requires_calculator", False)
    s_note = f"RAG retrieval · {'+ calculator' if calc else 'no calculator'} · conf {int(solution.get('confidence', 0)*100)}%"
    row("dot-pass", "3 · SolverAgent", s_note)

    # Verifier
    verdict = verification.get("verdict", "unknown")
    v_dot = "dot-pass" if verdict == "pass" else ("dot-fail" if verdict == "fail" else "dot-warn")
    v_note = f"verdict → {verdict}" + (f" · HITL triggered" if verification.get("needs_hitl") else "")
    row(v_dot, "4 · VerifierAgent", v_note)

    # Explainer
    e_dot = "dot-pass"
    e_note = f"difficulty → {explanation.get('difficulty', 'unknown')}"
    row(e_dot, "5 · ExplainerAgent", e_note)

    st.markdown("</div>", unsafe_allow_html=True)


def retrieved_context_panel(solution):
    docs = solution.get("retrieved_docs", [])
    sources = solution.get("sources_used", [])
    st.markdown('<div class="card"><div class="card-title">⬡ <span>Retrieved Context</span></div>', unsafe_allow_html=True)
    if not docs:
        st.markdown('<p style="color:var(--muted);font-size:0.83rem;">No context retrieved.</p>', unsafe_allow_html=True)
    for i, doc in enumerate(docs):
        label = sources[i] if i < len(sources) else f"Chunk {i+1}"
        st.markdown(f'<div class="chunk"><b style="color:var(--accent2);">{label}</b><br>{doc}</div>', unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


def solution_panel(solution, verification):
    # Confidence + verdict
    confidence_bar(solution.get("confidence", 0))
    v = verification.get("verdict", "unknown")
    badge_color = "#34d399" if v == "pass" else ("#f87171" if v == "fail" else "#fbbf24")
    st.markdown(f'<span style="background:{badge_color}22;color:{badge_color};border:1px solid {badge_color}55;border-radius:99px;padding:2px 10px;font-family:\'DM Mono\',monospace;font-size:0.72rem;">{v.upper()}</span>', unsafe_allow_html=True)

    # ── Step-by-step solution — clean notebook style ──
    steps = solution.get("solution", {}).get("steps", [])
    if steps:
        st.markdown('<div style="margin-top:1.4rem;"></div>', unsafe_allow_html=True)
        for step in steps:
            work_lines = latex_to_plain(step.get('work', '')).replace('\\n', '\n').split('\n')
            work_html = ''.join(
                f'<div style="font-size:1.05rem;font-family:Georgia,serif;color:#e2e8f0;padding:4px 0;line-height:1.9;">{line.strip()}</div>'
                for line in work_lines if line.strip()
            )
            st.markdown(f"""
            <div style="border-left:3px solid var(--accent2);padding:8px 0 8px 16px;margin-bottom:16px;">
              <div style="font-family:'DM Mono',monospace;font-size:0.68rem;color:var(--accent2);text-transform:uppercase;letter-spacing:1px;margin-bottom:8px;">Step {step.get('step_number')} · {step.get('description','')}</div>
              <div style="background:rgba(129,140,248,0.05);border-radius:8px;padding:14px 18px;">
                {work_html}
              </div>
            </div>""", unsafe_allow_html=True)
    else:
        # Fallback: show raw solver explanation if steps are missing
        fallback = solution.get("explanation", "") or solution.get("confidence_reason", "")
        if fallback:
            st.markdown(f'<p style="color:var(--muted);font-size:0.88rem;margin-top:1rem;">{latex_to_plain(fallback)}</p>', unsafe_allow_html=True)
        else:
            st.markdown(f'<p style="color:var(--danger);font-size:0.82rem;margin-top:1rem;">⚠ No solution steps returned. Raw result: {str(solution.get("solution", {}))[:200]}</p>', unsafe_allow_html=True)

    # Final answer box
    final = latex_to_plain(solution.get("solution", {}).get("final_answer", ""))
    if final:
        st.markdown(f"""
        <div class="answer-box">
          <div class="answer-label">Final Answer</div>
          <div class="answer-value">{final}</div>
        </div>""", unsafe_allow_html=True)

    # Formulas used (compact)
    formulas = solution.get("formulas_used", [])
    if formulas:
        st.markdown('<div style="font-family:\'DM Mono\',monospace;font-size:0.68rem;color:var(--muted);text-transform:uppercase;letter-spacing:1px;margin-top:1rem;margin-bottom:4px;">Formulas Used</div>', unsafe_allow_html=True)
        for f in formulas:
            st.markdown(f'<div style="font-family:Georgia,serif;font-size:0.92rem;color:var(--muted);padding:2px 0;">{latex_to_plain(f)}</div>', unsafe_allow_html=True)


def explanation_panel(explanation):
    """Show full student-friendly explanation."""
    concept = (explanation or {}).get("concept_intro", "")
    if concept:
        st.markdown(f'<p style="font-size:0.9rem;color:var(--muted);margin-bottom:1.2rem;line-height:1.7;">{concept}</p>', unsafe_allow_html=True)

    for step in (explanation or {}).get("steps", []):
        tip_html = f'<div class="step-tip">💡 {step["tip"]}</div>' if step.get("tip") else ""
        st.markdown(f"""
        <div class="step-block">
          <div class="step-num">Step {step['step_number']} · {step.get('heading','')}</div>
          <div class="step-exp">{step.get('explanation','')}</div>
          <div class="step-math">{latex_to_plain(step.get('math',''))}</div>
          {tip_html}
        </div>""", unsafe_allow_html=True)

    summary = (explanation or {}).get("summary", "")
    if summary:
        st.markdown(f'<p style="font-size:0.85rem;color:var(--muted);margin-top:0.8rem;line-height:1.6;">{summary}</p>', unsafe_allow_html=True)

    mistakes = (explanation or {}).get("common_mistakes", [])
    if mistakes:
        st.markdown('<div style="font-family:\'DM Mono\',monospace;font-size:0.68rem;color:var(--muted);text-transform:uppercase;letter-spacing:1px;margin-top:1rem;margin-bottom:6px;">Common Mistakes</div>', unsafe_allow_html=True)
        for m in mistakes:
            st.markdown(f'<div style="font-size:0.84rem;color:var(--danger);padding:3px 0;">✗ {m}</div>', unsafe_allow_html=True)

    follow_ups = (explanation or {}).get("follow_up_problems", [])
    if follow_ups:
        st.markdown('<div style="font-family:\'DM Mono\',monospace;font-size:0.68rem;color:var(--muted);text-transform:uppercase;letter-spacing:1px;margin-top:1rem;margin-bottom:6px;">Try These Next</div>', unsafe_allow_html=True)
        for p in follow_ups:
            st.markdown(f'<div style="font-size:0.84rem;color:var(--accent2);padding:3px 0;">→ {p}</div>', unsafe_allow_html=True)


def hitl_panel():
    """
    Shown when HITL is triggered. Lets a reviewer approve / edit / reject
    the solution before the student sees the explanation.
    """
    record = st.session_state.hitl_record
    if not record:
        return

    st.markdown(f"""
    <div class="hitl-banner">
      ⚠ <b>Human Review Required</b> — {st.session_state.hitl_reason}
      <span style="float:right;font-family:'DM Mono',monospace;font-size:0.7rem;color:var(--muted);">
        ID: {record.get('review_id')}
      </span>
    </div>""", unsafe_allow_html=True)

    st.markdown('<div class="card"><div class="card-title">⬡ <span>Reviewer Actions</span></div>', unsafe_allow_html=True)

    action = st.radio(
        "Action",
        ["✅ Approve — solution is correct", "✏️ Edit — provide corrected answer", "❌ Reject — solution is wrong, re-solve"],
        key="hitl_action_radio",
        label_visibility="collapsed",
    )

    if "Edit" in action:
        corrected = st.text_input("Corrected final answer", key="hitl_corrected")
        comment   = st.text_area("Explain the correction", key="hitl_comment", height=80)
    elif "Reject" in action:
        corrected = None
        comment   = st.text_area("Why is it wrong?", key="hitl_reject_comment", height=80)
    else:
        corrected = None
        comment   = ""

    if st.button("Submit Review", use_container_width=True):
        rid = record["review_id"]
        if "Approve" in action:
            updated = approve_review(rid)
            # Allow explainer to run
            st.session_state.verification["hitl_approved"] = True
        elif "Edit" in action:
            updated = edit_review(rid, corrected or "", comment)
            # Patch solution with corrected answer so explainer uses it
            if corrected:
                st.session_state.solution["solution"]["final_answer"] = corrected
            st.session_state.verification["hitl_approved"] = True
        else:
            updated = reject_review(rid, comment)
            st.session_state.verification["hitl_approved"] = False

        st.session_state.hitl_record   = updated
        st.session_state.hitl_resolved = True
        st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)


def feedback_section():
    st.markdown('<div class="card"><div class="card-title">⬡ <span>Student Feedback</span></div>', unsafe_allow_html=True)

    if st.session_state.feedback_given == "correct":
        st.markdown('<p style="color:var(--pass);font-size:0.88rem;">✓ Thanks — marked as correct.</p>', unsafe_allow_html=True)
    elif st.session_state.feedback_given == "incorrect":
        st.markdown('<p style="color:var(--danger);font-size:0.88rem;">✗ Marked as incorrect. Saved for review.</p>', unsafe_allow_html=True)
    else:
        col1, col2 = st.columns(2)
        with col1:
            if st.button("✅  Correct", use_container_width=True):
                save_student_feedback(
                    st.session_state.parsed,
                    st.session_state.solution,
                    feedback="correct",
                )
                if st.session_state.memory_id:
                    update_feedback(st.session_state.memory_id, "correct")
                st.session_state.feedback_given = "correct"
                st.rerun()
        with col2:
            if st.button("❌  Incorrect", use_container_width=True):
                st.session_state.show_comment_box = True
                st.rerun()

        if st.session_state.show_comment_box:
            comment   = st.text_area("What was wrong? (optional)", key="fb_comment", height=80)
            corrected = st.text_input("Correct answer (optional)", key="fb_corrected")
            if st.button("Submit Feedback"):
                save_student_feedback(
                    st.session_state.parsed,
                    st.session_state.solution,
                    feedback="incorrect",
                    comment=comment,
                )
                if st.session_state.memory_id:
                    update_feedback(
                        st.session_state.memory_id,
                        feedback="incorrect",
                        comment=comment,
                        corrected_answer=corrected,
                    )
                st.session_state.feedback_given  = "incorrect"
                st.session_state.feedback_comment = comment
                st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)

    # ── Request detailed explanation (manual HITL trigger) ──
    st.markdown('<div class="card"><div class="card-title">⬡ <span>Need More Help?</span></div>', unsafe_allow_html=True)
    if not st.session_state.hitl_triggered:
        if st.button("🔍  Request Detailed Explanation", use_container_width=True):
            st.session_state.hitl_triggered = True
            st.session_state.hitl_reason    = "Student requested a more detailed explanation."
            st.session_state.hitl_record    = create_review(
                st.session_state.parsed,
                st.session_state.routing,
                st.session_state.solution,
                st.session_state.verification,
                "Student requested a more detailed explanation.",
            )
            st.rerun()
    else:
        st.markdown('<p style="color:var(--accent);font-size:0.85rem;">✓ Detailed explanation request submitted.</p>', unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


def memory_panel():
    """Show similar problems retrieved from memory + live stats."""
    similar = st.session_state.get("similar_problems", [])
    stats   = get_stats()

    st.markdown('<div class="card"><div class="card-title">⬡ <span>Memory</span></div>', unsafe_allow_html=True)

    # Stats row
    st.markdown(f"""
    <div style="display:flex;gap:20px;margin-bottom:0.8rem;">
      <div style="text-align:center;">
        <div style="font-family:'DM Mono',monospace;font-size:1.1rem;color:var(--accent);">
          {stats.get('total_solved', 0)}
        </div>
        <div style="font-size:0.7rem;color:var(--muted);">problems stored</div>
      </div>
      <div style="text-align:center;">
        <div style="font-family:'DM Mono',monospace;font-size:1.1rem;color:var(--pass);">
          {int(stats.get('correct_rate', 0) * 100)}%
        </div>
        <div style="font-size:0.7rem;color:var(--muted);">accuracy</div>
      </div>
      <div style="text-align:center;">
        <div style="font-family:'DM Mono',monospace;font-size:1.1rem;color:var(--accent2);">
          {stats.get('correction_rules', 0)}
        </div>
        <div style="font-size:0.7rem;color:var(--muted);">OCR/ASR rules</div>
      </div>
    </div>""", unsafe_allow_html=True)

    if similar:
        st.markdown('<div style="font-family:\'DM Mono\',monospace;font-size:0.7rem;color:var(--muted);text-transform:uppercase;letter-spacing:1px;margin-bottom:6px;">Similar Problems Used</div>', unsafe_allow_html=True)
        for r in similar:
            answer = r.get("corrected_answer") or r.get("final_answer", "")
            fb_color = "var(--pass)" if r.get("feedback") == "correct" else (
                "var(--danger)" if r.get("feedback") == "incorrect" else "var(--muted)"
            )
            st.markdown(f"""
            <div class="chunk">
              <span style="color:var(--accent2);">{r.get('topic','').upper()}</span>
              &nbsp;·&nbsp;
              <span style="color:{fb_color};font-size:0.7rem;">{r.get('feedback') or r.get('verdict','')}</span>
              <br>{r.get('problem_text','')[:120]}{'…' if len(r.get('problem_text',''))>120 else ''}
              <br><span style="color:var(--accent);font-family:'DM Mono',monospace;">→ {answer}</span>
            </div>""", unsafe_allow_html=True)
    else:
        st.markdown('<p style="color:var(--muted);font-size:0.82rem;">No similar problems found in memory yet.</p>', unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)


# ── Main layout ───────────────────────────────────────────────────────────────
st.markdown("""
<div class="mentor-header">
  <h1 class="mentor-title">∑ Math Mentor</h1>
  <span class="mentor-sub">JEE · RAG + Agents · v1.0</span>
</div>""", unsafe_allow_html=True)

left, right = st.columns([1, 1.4], gap="large")

# ── LEFT COLUMN — Input ───────────────────────────────────────────────────────
with left:
    # Mode selector
    mode = st.radio(
        "Input mode",
        ["Text", "Image", "Audio"],
        horizontal=True,
        label_visibility="collapsed",
        key="input_mode_radio",
    )
    st.session_state.input_mode = mode
    st.markdown("---")

    raw_text = ""

    if mode == "Text":
        raw_text = st.text_area(
            "Type your math problem",
            placeholder="e.g. If X and Y are independent with P(X)=0.3 and P(Y)=0.5, find P(X∪Y)",
            height=140,
            label_visibility="collapsed",
        )
        st.session_state.raw_input = raw_text

    elif mode == "Audio":
        audio_bytes = None
        audio_mime = "audio/wav"

        # Try mic recorder component
        recorder_available = False
        _mic_recorder = None
        _audio_recorder = None
        try:
            from audio_recorder_streamlit import audio_recorder as _audio_recorder
            recorder_available = True
        except ImportError:
            try:
                from streamlit_mic_recorder import mic_recorder as _mic_recorder
                recorder_available = True
            except ImportError:
                pass

        if recorder_available and _audio_recorder:
            st.markdown('<p style="color:var(--muted);font-size:0.82rem;">Click the mic to start recording, click again to stop.</p>', unsafe_allow_html=True)
            recorded = _audio_recorder(
                text="",
                recording_color="#f87171",
                neutral_color="#6ee7b7",
                icon_name="microphone",
                icon_size="3x",
                pause_threshold=3.0,
                key="audio_recorder",
            )
            if recorded:
                audio_bytes = recorded

        elif recorder_available and _mic_recorder:
            st.markdown('<p style="color:var(--muted);font-size:0.82rem;">Click the mic to start, click again to stop.</p>', unsafe_allow_html=True)
            audio = _mic_recorder(
                start_prompt="🎙️  Start Recording",
                stop_prompt="⏹️  Stop Recording",
                just_once=False,
                key="mic_recorder",
            )
            if audio:
                audio_bytes = audio["bytes"]

        else:
            st.warning("Install mic recorder:  pip install streamlit-mic-recorder")

        # Fallback file upload
        st.markdown('<p style="color:var(--muted);font-size:0.75rem;text-align:center;margin:4px 0;">— or upload an audio file —</p>', unsafe_allow_html=True)
        audio_file = st.file_uploader("Upload audio", type=["wav", "mp3", "m4a"], label_visibility="collapsed")
        if audio_file:
            audio_bytes = audio_file.read()
            ext = audio_file.name.split(".")[-1].lower()
            audio_mime = {"wav": "audio/wav", "mp3": "audio/mpeg", "m4a": "audio/mp4"}.get(ext, "audio/wav")

        if audio_bytes and st.button("🔊 Transcribe", use_container_width=True):
            with st.spinner("Transcribing…"):
                try:
                    import os, tempfile, pathlib
                    from dotenv import load_dotenv
                    load_dotenv()
                    import google.generativeai as genai_audio
                    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
                    genai_audio.configure(api_key=api_key)
                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                        tmp.write(audio_bytes)
                        tmp_path = tmp.name
                    uploaded_audio = genai_audio.upload_file(tmp_path, mime_type=audio_mime)
                    asr_model = genai_audio.GenerativeModel("gemini-2.5-flash")
                    asr_response = asr_model.generate_content([
                        uploaded_audio,
                        (
                            "Transcribe this audio. The speaker is stating a math problem.\n"
                            "Write math in plain readable notation — NOT LaTeX.\n"
                            "Convert spoken math: 'x squared' → x², 'square root of x' → sqrt(x), "
                            "'integral from a to b' → integral from a to b of, 'x over y' → x/y.\n"
                            "Output only the transcribed math problem, nothing else."
                        )
                    ])
                    pathlib.Path(tmp_path).unlink(missing_ok=True)
                    st.session_state.extracted_text = asr_response.text.strip()
                except Exception as asr_err:
                    st.warning(f"Transcription failed: {asr_err}")

        if st.session_state.get("extracted_text") and mode == "Audio":
            st.markdown('<div class="card"><div class="card-title">⬡ <span>Transcript — edit if needed</span></div>', unsafe_allow_html=True)
            transcript = st.text_area(
                "Transcript",
                value=st.session_state.extracted_text,
                height=100,
                label_visibility="collapsed",
            )
            st.session_state.extracted_text = transcript
            raw_text = transcript
            st.markdown("</div>", unsafe_allow_html=True)
        st.session_state.raw_input = raw_text

    elif mode == "Image":
        uploaded = st.file_uploader("Upload image (JPG / PNG)", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
        if uploaded:
            st.image(uploaded, use_container_width=True)

            # Run OCR automatically when a new image is uploaded
            img_key = f"ocr_{uploaded.name}_{uploaded.size}"
            if st.session_state.get("last_ocr_key") != img_key:
                with st.spinner("Extracting text from image…"):
                    try:
                        import base64, os
                        from dotenv import load_dotenv
                        load_dotenv()
                        from langchain_google_genai import ChatGoogleGenerativeAI
                        from langchain_core.messages import HumanMessage
                        api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
                        img_bytes = uploaded.read()
                        b64 = base64.b64encode(img_bytes).decode("utf-8")
                        ext = uploaded.name.split(".")[-1].lower()
                        mime = "image/png" if ext == "png" else "image/jpeg"
                        ocr_model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0, google_api_key=api_key)
                        ocr_response = ocr_model.invoke([
                            HumanMessage(content=[
                                {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
                                {"type": "text", "text": (
                                    "Extract all text and math from this image in plain readable notation — NOT LaTeX.\n"
                                    "Rules:\n"
                                    "- Write fractions as: a/b or (a+b)/(c+d)\n"
                                    "- Write powers as: x², x³, x^n\n"
                                    "- Write derivatives as: d/dx, f'(x), dy/dx\n"
                                    "- Write integrals as: integral from a to b of f(t)dt\n"
                                    "- Write square roots as: sqrt(x)\n"
                                    "- Do NOT use $, \\frac, \\int, \\sqrt, ^ or any LaTeX commands\n"
                                    "Output only the extracted text, nothing else."
                                )},
                            ])
                        ])
                        import re as _re
                        raw = ocr_response.content.strip()
                        # Strip any LaTeX that slipped through
                        raw = _re.sub(r'\$+', '', raw)
                        raw = _re.sub(r'\\frac\{([^}]+)\}\{([^}]+)\}', r'(\1)/(\2)', raw)
                        raw = _re.sub(r'\\sqrt\{([^}]+)\}', r'sqrt(\1)', raw)
                        raw = _re.sub(r'\\int_\{([^}]+)\}\^\{([^}]+)\}', r'integral from \1 to \2 of', raw)
                        raw = _re.sub(r'\\int', 'integral', raw)
                        raw = _re.sub(r'\\sum_\{([^}]+)\}\^\{([^}]+)\}', r'sum from \1 to \2 of', raw)
                        raw = _re.sub(r'\\sum', 'sum', raw)
                        raw = _re.sub(r'\\left|\\right|\\cdot|\\times|\\pm', lambda m: {'\\cdot': '×', '\\times': '×', '\\pm': '±'}.get(m.group(), ' '), raw)
                        raw = _re.sub(r'\\\w+', '', raw)
                        raw = _re.sub(r'[{}]', '', raw)
                        raw = _re.sub(r' +', ' ', raw).strip()
                        st.session_state.extracted_text = raw
                        st.session_state.last_ocr_key = img_key
                    except Exception as ocr_err:
                        st.session_state.extracted_text = ""
                        st.warning(f"OCR failed: {ocr_err}")

            st.markdown('<div class="card"><div class="card-title">⬡ <span>OCR Preview — edit if needed</span></div>', unsafe_allow_html=True)
            extracted = st.text_area(
                "Extracted text",
                value=st.session_state.extracted_text or "",
                height=100,
                label_visibility="collapsed",
            )
            st.session_state.extracted_text = extracted
            raw_text = extracted
            st.markdown("</div>", unsafe_allow_html=True)
        st.session_state.raw_input = raw_text

    st.markdown("<br>", unsafe_allow_html=True)

    solve_clicked = st.button("→ Solve", use_container_width=True)

    if solve_clicked:
        if not raw_text or not raw_text.strip():
            st.warning("Please enter a math problem first.")
        else:
            # Reset state
            for key in ["parsed", "routing", "solution", "verification", "explanation",
                        "pipeline_ran", "feedback_given", "feedback_comment", "show_comment_box",
                        "hitl_triggered", "hitl_reason", "hitl_record", "hitl_resolved",
                        "memory_id", "similar_problems", "want_explanation"]:
                st.session_state[key] = None if key not in [
                    "pipeline_ran", "show_comment_box", "hitl_triggered", "hitl_resolved",
                    "similar_problems"
                ] else (False if key not in ["similar_problems"] else [])

            with st.spinner("Running agents…"):
                try:
                    result = run_pipeline(raw_text, mode)

                    st.session_state.parsed       = result["parsed"]
                    st.session_state.routing      = result["routing"]
                    st.session_state.solution     = result["solution"]
                    st.session_state.verification = result["verification"]
                    st.session_state.explanation  = result["explanation"]
                    st.session_state.memory_id    = result.get("memory_id")
                    st.session_state.similar_problems = result.get("similar_problems", [])
                    st.session_state.pipeline_ran = True

                    # ── HITL only triggered manually by user, not automatically ──

                except Exception as e:
                    st.error(f"Pipeline error: {e}")

    # Agent trace (always in left column once run)
    if st.session_state.pipeline_ran:
        st.markdown("<br>", unsafe_allow_html=True)
        agent_trace(
            st.session_state.parsed,
            st.session_state.routing,
            st.session_state.solution,
            st.session_state.verification,
            st.session_state.explanation,
        )


# ── RIGHT COLUMN — Results ────────────────────────────────────────────────────
with right:
    if not st.session_state.pipeline_ran:
        st.markdown("""
        <div style="display:flex;flex-direction:column;align-items:center;justify-content:center;
                    height:60vh;color:var(--muted);text-align:center;gap:12px;">
          <div style="font-size:3rem;opacity:0.3;">∑</div>
          <div style="font-family:'DM Mono',monospace;font-size:0.78rem;">
            Enter a problem and hit Solve
          </div>
        </div>""", unsafe_allow_html=True)
    else:
        sol  = st.session_state.solution
        ver  = st.session_state.verification
        exp  = st.session_state.explanation
        rout = st.session_state.routing

        # ── HITL panel (shown before solution if triggered and not yet resolved)
        if st.session_state.hitl_triggered and not st.session_state.hitl_resolved:
            hitl_panel()
            st.markdown("---")

        # ── If HITL was rejected, block the solution view
        hitl_record = st.session_state.hitl_record or {}
        if hitl_record.get("human_action") == "reject":
            st.markdown("""
            <div class="hitl-banner">
              ❌ <b>Solution Rejected by Reviewer</b> — This problem has been flagged for re-solving.
              Please re-submit with a corrected or clearer problem statement.
            </div>""", unsafe_allow_html=True)
        else:
            # Main solution panel
            st.markdown('<div class="card"><div class="card-title">⬡ <span>Solution</span></div>', unsafe_allow_html=True)
            solution_panel(sol, ver)
            st.markdown("</div>", unsafe_allow_html=True)

            # ── Explanation on demand ──────────────────────────────────────
            if not st.session_state.want_explanation:
                st.markdown('<div style="margin-top:0.5rem;"></div>', unsafe_allow_html=True)
                col1, col2 = st.columns([2, 1])
                with col1:
                    st.markdown('<p style="color:var(--muted);font-size:0.88rem;margin:0;padding-top:8px;">Do you want a detailed explanation of this solution?</p>', unsafe_allow_html=True)
                with col2:
                    if st.button("✦ Explain This", use_container_width=True):
                        st.session_state.want_explanation = True
                        st.rerun()
            else:
                st.markdown('<div class="card"><div class="card-title">⬡ <span>Explanation</span></div>', unsafe_allow_html=True)
                if exp and exp.get("steps"):
                    explanation_panel(exp)
                else:
                    st.markdown('<p style="color:var(--muted);font-size:0.85rem;">No explanation available for this solution.</p>', unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)