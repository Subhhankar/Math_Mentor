"""
memory.py — Memory & Self-Learning Layer for Math Mentor

Stores every solved problem and retrieves similar ones at runtime.
Uses sentence-transformers for semantic similarity (no API call needed).
Falls back to keyword overlap if the model isn't available.

Storage: memory_store.jsonl  (one JSON record per line)
"""

import json
import uuid
import datetime
import re
from pathlib import Path
from typing import Optional

MEMORY_FILE = Path("memory_store.jsonl")

# ── Optional: semantic similarity ─────────────────────────────────────────────
try:
    from sentence_transformers import SentenceTransformer, util
    import torch
    _MODEL = SentenceTransformer("all-MiniLM-L6-v2")
    _SEMANTIC = True
except ImportError:
    _SEMANTIC = False


# ═══════════════════════════════════════════════════════════════════════════════
# 1. SAVE  —  called after every successful pipeline run
# ═══════════════════════════════════════════════════════════════════════════════

def save_memory(
    raw_input: str,
    input_source: str,          # "text" | "ocr" | "asr"
    parsed: dict,
    solution: dict,
    verification: dict,
    retrieved_docs: list[str],
    feedback: Optional[str] = None,      # "correct" | "incorrect" | None
    feedback_comment: Optional[str] = None,
    corrected_answer: Optional[str] = None,
) -> dict:
    """
    Persist a complete solved-problem record to disk.
    Returns the saved record (including its memory_id).
    """
    record = {
        "memory_id":        str(uuid.uuid4())[:10],
        "timestamp":        datetime.datetime.utcnow().isoformat(),

        # ── Input ──────────────────────────────────────────────────────────
        "raw_input":        raw_input,
        "input_source":     input_source,

        # ── Parsed problem ─────────────────────────────────────────────────
        "problem_text":     parsed.get("problem_text", ""),
        "topic":            parsed.get("topic", "other"),
        "variables":        parsed.get("variables", []),
        "constraints":      parsed.get("constraints", []),

        # ── RAG context ────────────────────────────────────────────────────
        "retrieved_docs":   retrieved_docs,
        "sources_used":     solution.get("sources_used", []),

        # ── Solution ───────────────────────────────────────────────────────
        "final_answer":     solution.get("solution", {}).get("final_answer", ""),
        "answer_latex":     solution.get("solution", {}).get("answer_latex", ""),
        "steps":            solution.get("solution", {}).get("steps", []),
        "formulas_used":    solution.get("formulas_used", []),
        "confidence":       solution.get("confidence", 0.0),

        # ── Verification ───────────────────────────────────────────────────
        "verdict":          verification.get("verdict", "unknown"),
        "verifier_issues":  (
            verification.get("correctness", {}).get("issues", []) +
            verification.get("units_and_domain", {}).get("issues", []) +
            verification.get("edge_cases", {}).get("issues", [])
        ),

        # ── Feedback ───────────────────────────────────────────────────────
        "feedback":         feedback,
        "feedback_comment": feedback_comment,
        "corrected_answer": corrected_answer,

        # ── OCR / ASR correction rules ─────────────────────────────────────
        # If OCR/ASR was used AND user corrected extracted text, store the
        # raw→clean mapping so future inputs benefit from it.
        "ocr_asr_correction": None,
    }

    _append(record)
    return record


def save_ocr_asr_correction(raw_extracted: str, corrected_text: str, source: str):
    """
    Save a raw→clean correction pair from the OCR/ASR preview edit box.
    Used by apply_ocr_asr_corrections() at runtime.
    """
    record = {
        "memory_id":    str(uuid.uuid4())[:10],
        "timestamp":    datetime.datetime.utcnow().isoformat(),
        "type":         "ocr_asr_correction",
        "source":       source,
        "raw":          raw_extracted,
        "corrected":    corrected_text,
    }
    _append(record)
    return record


# ═══════════════════════════════════════════════════════════════════════════════
# 2. RETRIEVE  —  called before SolverAgent runs
# ═══════════════════════════════════════════════════════════════════════════════

def retrieve_similar(
    problem_text: str,
    topic: str,
    top_k: int = 3,
    min_confidence: float = 0.75,
    only_verified: bool = True,
) -> list[dict]:
    """
    Find the most similar previously solved problems.

    Priority order:
      1. Same topic + verified (verdict=pass) + positive feedback
      2. Same topic + verified
      3. Semantic / keyword similarity across all records

    Returns up to top_k records, formatted as memory context strings.
    """
    all_records = _load_solved()

    # Filter: skip low-confidence and (optionally) unverified
    candidates = [
        r for r in all_records
        if r.get("confidence", 0) >= min_confidence
        and (not only_verified or r.get("verdict") == "pass")
    ]

    if not candidates:
        return []

    # Topic boost: put same-topic records first
    same_topic   = [r for r in candidates if r.get("topic") == topic]
    other_topics = [r for r in candidates if r.get("topic") != topic]
    ordered = same_topic + other_topics

    # Score by similarity
    if _SEMANTIC:
        scored = _semantic_rank(problem_text, ordered)
    else:
        scored = _keyword_rank(problem_text, ordered)

    return scored[:top_k]


def format_memory_context(similar_records: list[dict]) -> str:
    """
    Format retrieved memory records into a string for injection into
    the SolverAgent prompt (memory_context parameter).
    """
    if not similar_records:
        return "No similar problems found in memory."

    lines = ["=== SIMILAR SOLVED PROBLEMS FROM MEMORY ===\n"]
    for i, r in enumerate(similar_records, 1):
        # Use corrected answer if human edited it
        answer = r.get("corrected_answer") or r.get("final_answer", "")
        lines.append(
            f"[{i}] Problem : {r.get('problem_text', '')}\n"
            f"    Topic   : {r.get('topic', '')}\n"
            f"    Answer  : {answer}\n"
            f"    Formulas: {', '.join(r.get('formulas_used', []))}\n"
            f"    Verdict : {r.get('verdict', '')}  |  "
            f"Feedback: {r.get('feedback') or 'none'}\n"
        )
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# 3. OCR / ASR CORRECTION RULES
# ═══════════════════════════════════════════════════════════════════════════════

def apply_ocr_asr_corrections(raw_text: str, source: str) -> str:
    """
    Apply known OCR/ASR correction rules learned from past human edits.
    Called in the parser before the LLM sees the text.
    """
    corrections = _load_corrections(source)
    text = raw_text

    for rule in corrections:
        raw_pattern = rule.get("raw", "")
        clean       = rule.get("corrected", "")
        if raw_pattern and raw_pattern in text:
            text = text.replace(raw_pattern, clean)

    return text


def get_correction_rules(source: str) -> list[dict]:
    """Return all stored OCR/ASR correction rules for a given source."""
    return _load_corrections(source)


# ═══════════════════════════════════════════════════════════════════════════════
# 4. SOLUTION PATTERN REUSE
# ═══════════════════════════════════════════════════════════════════════════════

def get_solution_pattern(topic: str, sub_type: str) -> Optional[dict]:
    """
    Return the most recent verified+approved solution pattern for a
    given topic/sub_type combination.
    Used by SolverAgent to hint at the correct approach.
    """
    all_records = _load_solved()
    matches = [
        r for r in all_records
        if r.get("topic") == topic
        and r.get("verdict") == "pass"
        and r.get("feedback") in ("correct", None)
        and r.get("formulas_used")
    ]
    if not matches:
        return None
    # Most recent first
    matches.sort(key=lambda r: r.get("timestamp", ""), reverse=True)
    best = matches[0]
    return {
        "formulas_used":  best.get("formulas_used", []),
        "example_steps":  best.get("steps", [])[:2],   # first 2 steps as a hint
        "example_answer": best.get("corrected_answer") or best.get("final_answer"),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 5. FEEDBACK UPDATE  —  called when student clicks ✅ / ❌
# ═══════════════════════════════════════════════════════════════════════════════

def update_feedback(
    memory_id: str,
    feedback: str,
    comment: str = "",
    corrected_answer: str = "",
):
    """Patch an existing memory record with student feedback."""
    records = _load_all()
    for r in records:
        if r.get("memory_id") == memory_id:
            r["feedback"]         = feedback
            r["feedback_comment"] = comment
            if corrected_answer:
                r["corrected_answer"] = corrected_answer
            break
    _rewrite(records)


# ═══════════════════════════════════════════════════════════════════════════════
# 6. STATS  —  useful for the UI / debugging
# ═══════════════════════════════════════════════════════════════════════════════

def get_stats() -> dict:
    records = _load_solved()
    total   = len(records)
    by_topic = {}
    correct  = 0
    for r in records:
        t = r.get("topic", "other")
        by_topic[t] = by_topic.get(t, 0) + 1
        if r.get("feedback") == "correct" or r.get("verdict") == "pass":
            correct += 1
    return {
        "total_solved":   total,
        "by_topic":       by_topic,
        "correct_rate":   round(correct / total, 2) if total else 0,
        "correction_rules": len(_load_all_corrections()),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Internal helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _append(record: dict):
    with open(MEMORY_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def _load_all() -> list[dict]:
    if not MEMORY_FILE.exists():
        return []
    out = []
    with open(MEMORY_FILE, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return out


def _rewrite(records: list[dict]):
    with open(MEMORY_FILE, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def _load_solved() -> list[dict]:
    """Return only full solved-problem records (not correction rules)."""
    return [r for r in _load_all() if r.get("type") != "ocr_asr_correction"]


def _load_corrections(source: str) -> list[dict]:
    return [
        r for r in _load_all()
        if r.get("type") == "ocr_asr_correction"
        and r.get("source") == source
    ]


def _load_all_corrections() -> list[dict]:
    return [r for r in _load_all() if r.get("type") == "ocr_asr_correction"]


def _keyword_rank(query: str, records: list[dict]) -> list[dict]:
    """Simple token overlap scoring — fallback when sentence-transformers unavailable."""
    query_tokens = set(re.findall(r"\w+", query.lower()))
    scored = []
    for r in records:
        text   = r.get("problem_text", "").lower()
        tokens = set(re.findall(r"\w+", text))
        overlap = len(query_tokens & tokens) / max(len(query_tokens), 1)
        scored.append((overlap, r))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [r for _, r in scored if _ > 0]


def _semantic_rank(query: str, records: list[dict]) -> list[dict]:
    """Cosine similarity using sentence-transformers."""
    texts = [r.get("problem_text", "") for r in records]
    if not texts:
        return []
    query_emb  = _MODEL.encode(query, convert_to_tensor=True)
    corpus_emb = _MODEL.encode(texts, convert_to_tensor=True)
    scores     = util.cos_sim(query_emb, corpus_emb)[0].tolist()
    ranked     = sorted(zip(scores, records), key=lambda x: x[0], reverse=True)
    return [r for score, r in ranked if score > 0.3]


# ── Quick test ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Save a mock record
    mock_parsed = {
        "problem_text": "If P(A)=0.3 and P(B)=0.5 and A,B independent, find P(A∪B)",
        "topic": "probability",
        "variables": ["A", "B"],
        "constraints": ["P(A)=0.3", "P(B)=0.5"],
    }
    mock_solution = {
        "solution": {
            "final_answer": "0.65",
            "answer_latex": "0.65",
            "steps": [{"step_number": 1, "description": "Apply union rule",
                        "work": "0.3+0.5-0.15=0.65", "reasoning": "inclusion-exclusion"}],
        },
        "formulas_used": ["P(A∪B)=P(A)+P(B)-P(A∩B)"],
        "confidence": 0.93,
        "sources_used": ["Probability Sheet"],
    }
    mock_verification = {"verdict": "pass", "correctness": {"issues": []},
                         "units_and_domain": {"issues": []}, "edge_cases": {"issues": []}}

    record = save_memory(
        raw_input="If P(A)=0.3 and P(B)=0.5, independent, find P(A∪B)",
        input_source="text",
        parsed=mock_parsed,
        solution=mock_solution,
        verification=mock_verification,
        retrieved_docs=["For independent events P(A∩B)=P(A)·P(B)"],
        feedback="correct",
    )
    print("Saved:", record["memory_id"])

    # Retrieve similar
    similar = retrieve_similar("find P(X union Y) for independent events X and Y", topic="probability")
    print("\nRetrieved:", len(similar), "similar problems")
    print(format_memory_context(similar))

    print("\nStats:", get_stats())