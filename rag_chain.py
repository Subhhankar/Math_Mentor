"""
rag_chain.py
------------
Gemini-first RAG chain for the math agent.

Architecture:
    1. Gemini solves the problem independently (primary)
    2. If confidence < 0.75, RAG context is fetched and used as a hint
    3. Pinecone knowledge base is a reference library, NOT the answer source

Usage:
    from rag_chain import solve

    result = solve("Find the probability that sum of two dice is 4 or 5")
    print(result["final_answer"])
    print(result["confidence"])
"""

import os
import json
import re
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from vector_store import get_retriever

# ──────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────

load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

CONFIDENCE_THRESHOLD = 0.75   # below this → enrich with RAG context

# ──────────────────────────────────────────────
# Gemini model (singleton)
# ──────────────────────────────────────────────

gemini = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.3,
    max_tokens=2048,
)

# ──────────────────────────────────────────────
# Prompts
# ──────────────────────────────────────────────

# ── Stage 1: Gemini solves alone ─────────────
DIRECT_SYSTEM = """You are an expert mathematics tutor specialised in JEE-level problems
(Algebra, Probability, Calculus, Linear Algebra).

Solve the problem below using your own reasoning. Do NOT wait for external context.

Respond ONLY in this strict JSON format (no markdown, no backticks):

{{
  "solution": {{
    "steps": [
      {{
        "step_number": 1,
        "description": "Brief label",
        "work": "Detailed mathematical working",
        "reasoning": "Why this step is taken"
      }}
    ],
    "final_answer": "Final answer with units/domain if applicable",
    "answer_latex": "LaTeX representation of the final answer"
  }},
  "explanation": "2-4 sentence student-friendly explanation of the approach",
  "topic": "Math topic this problem belongs to",
  "formulas_used": ["Formulas or identities applied"],
  "confidence": 0.0,
  "confidence_reason": "Why you are or are not fully confident",
  "edge_cases": "Domain restrictions, special cases, or caveats",
  "needs_human_review": false,
  "sources_used": []
}}

RULES:
- confidence is a float 0.0 – 1.0 reflecting how certain you are of your answer
- If confidence < 0.75, set needs_human_review to true and add review_reason field
- Show ALL algebraic steps; never skip manipulations
- Use exact mathematical notation in work fields
"""

DIRECT_PROMPT = ChatPromptTemplate.from_messages([
    ("system", DIRECT_SYSTEM),
    ("human", "Problem: {problem}\n\nAdditional context: {memory_context}"),
])

# ── Stage 2: Gemini + RAG context hint ───────
RAG_SYSTEM = """You are an expert mathematics tutor specialised in JEE-level problems
(Algebra, Probability, Calculus, Linear Algebra).

You have already attempted this problem. Below is supplementary reference material
retrieved from a knowledge base. Use it ONLY to verify your reasoning or fill in
specific formula references you were uncertain about.

If the context contradicts your own correct reasoning, trust your reasoning.
The context is a hint — not the answer.

---
REFERENCE CONTEXT (use only if helpful):
{context}
---

Respond ONLY in this strict JSON format (no markdown, no backticks):

{{
  "solution": {{
    "steps": [
      {{
        "step_number": 1,
        "description": "Brief label",
        "work": "Detailed mathematical working",
        "reasoning": "Why this step is taken"
      }}
    ],
    "final_answer": "Final answer with units/domain if applicable",
    "answer_latex": "LaTeX representation of the final answer"
  }},
  "explanation": "2-4 sentence student-friendly explanation of the approach",
  "topic": "Math topic this problem belongs to",
  "formulas_used": ["Formulas or identities applied"],
  "confidence": 0.0,
  "confidence_reason": "Why you are or are not fully confident",
  "edge_cases": "Domain restrictions, special cases, or caveats",
  "needs_human_review": false,
  "review_reason": "",
  "sources_used": ["Only list context chunks you actually used"]
}}

RULES:
- confidence is a float 0.0 – 1.0
- If confidence < 0.75, set needs_human_review to true
- Do NOT hallucinate formulas or cite sources not in the context above
- Show ALL steps; never skip algebraic manipulations
"""

RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system", RAG_SYSTEM),
    ("human", "Problem: {problem}\n\nAdditional context: {memory_context}"),
])


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def _parse_json(raw: str) -> dict:
    """Safely parse JSON from Gemini output, stripping markdown fences."""
    cleaned = re.sub(r"```(?:json)?|```", "", raw).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        # Return raw text wrapped so callers always get a dict
        return {
            "final_answer": raw,
            "confidence": 0.0,
            "confidence_reason": "Failed to parse structured response",
            "needs_human_review": True,
            "parse_error": True,
        }


def _extract_confidence(parsed: dict) -> float:
    return float(parsed.get("confidence", 0.0))


def _format_docs(docs) -> str:
    if not docs:
        return "No relevant context found."
    return "\n\n---\n\n".join(
        f"[Source: {d.metadata.get('source','unknown')} | "
        f"Topic: {d.metadata.get('topic','unknown')}]\n{d.page_content}"
        for d in docs
    )


# ──────────────────────────────────────────────
# Core solver
# ──────────────────────────────────────────────

def solve(
    problem: str,
    memory_context: str = "",
    force_rag: bool = False,
    k: int = 3,
) -> dict:
    """
    Gemini-first solver with optional RAG fallback.

    Args:
        problem:        The math problem as a plain string.
        memory_context: Similar previously-solved problems (from memory agent).
        force_rag:      Set True to always use RAG (e.g. for niche formula lookup).
        k:              Number of docs to retrieve if RAG is triggered.

    Returns:
        Parsed solution dict with keys: solution, final_answer, confidence, etc.
    """

    # ── Stage 1: Gemini solves independently ─
    print("[rag_chain] Stage 1: Gemini solving independently...")
    direct_chain = DIRECT_PROMPT | gemini | StrOutputParser()
    raw_direct   = direct_chain.invoke({
        "problem":        problem,
        "memory_context": memory_context,
    })
    result = _parse_json(raw_direct)
    confidence = _extract_confidence(result)
    result["rag_used"] = False

    print(f"[rag_chain] Gemini confidence: {confidence:.2f}")

    # ── Stage 2: RAG enrichment if needed ────
    if force_rag or confidence < CONFIDENCE_THRESHOLD:
        print(f"[rag_chain] Stage 2: Confidence {confidence:.2f} < {CONFIDENCE_THRESHOLD}. "
              "Fetching RAG context...")

        retriever = get_retriever(k=k)
        docs      = retriever.get_relevant_documents(problem)
        context   = _format_docs(docs)

        rag_chain = RAG_PROMPT | gemini | StrOutputParser()
        raw_rag   = rag_chain.invoke({
            "problem":        problem,
            "memory_context": memory_context,
            "context":        context,
        })
        result = _parse_json(raw_rag)
        result["rag_used"]      = True
        result["retrieved_docs"] = [
            {"source": d.metadata.get("source"), "topic": d.metadata.get("topic")}
            for d in docs
        ]
        print(f"[rag_chain] RAG confidence: {_extract_confidence(result):.2f}")
    else:
        print("[rag_chain] Gemini confident. Skipping RAG retrieval.")

    return result


# ──────────────────────────────────────────────
# Quick test  (python rag_chain.py)
# ──────────────────────────────────────────────

if __name__ == "__main__":
    test_problem = (
        "One die has two faces marked 1, two faces marked 2, one face marked 3 "
        "and one face marked 4. Another die has one face marked 1, two faces marked 2, "
        "two faces marked 3 and one face marked 4. "
        "Find the probability of getting a sum of 4 or 5 when both dice are thrown."
    )

    result = solve(test_problem)

    print("\n" + "=" * 70)
    print("SOLUTION")
    print("=" * 70)
    print(f"Final Answer : {result.get('final_answer')}")
    print(f"Confidence   : {result.get('confidence')}")
    print(f"RAG used     : {result.get('rag_used')}")
    print(f"Human review : {result.get('needs_human_review')}")
    print(f"\nExplanation  : {result.get('explanation')}")