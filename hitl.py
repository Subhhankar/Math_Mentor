import json
import uuid
import datetime
from pathlib import Path


HITL_STORE = Path("hitl_reviews.jsonl")   # flat file — swap for DB if needed


# ── Trigger detection ─────────────────────────────────────────────────────────

def should_trigger_hitl(parsed: dict, verification: dict, solution: dict) -> tuple[bool, str]:
    """
    Central check — returns (trigger: bool, reason: str).
    Call this after every pipeline run before showing the explanation.
    """
    # 1. Parser detected ambiguity
    if parsed.get("needs_clarification"):
        return True, f"Parser ambiguity: {parsed.get('clarification_reason', 'unclear input')}"

    # 2. Verifier is not confident
    if verification.get("needs_hitl"):
        return True, f"Verifier: {verification.get('hitl_reason', 'low confidence')}"

    # 3. Solver itself flagged low confidence
    if solution.get("needs_human_review"):
        return True, f"Solver low confidence: {solution.get('review_reason', '')}"

    # 4. Verifier verdict is not pass
    if verification.get("verdict") in ("fail", "uncertain"):
        return True, f"Verifier verdict: {verification.get('verdict')}"

    return False, ""


# ── Review record ─────────────────────────────────────────────────────────────

def create_review(
    parsed: dict,
    routing: dict,
    solution: dict,
    verification: dict,
    trigger_reason: str,
) -> dict:
    """
    Build a HITL review record and persist it to disk.
    Returns the record (including its ID) for use in the UI.
    """
    record = {
        "review_id":      str(uuid.uuid4())[:8],
        "timestamp":      datetime.datetime.utcnow().isoformat(),
        "status":         "pending",          # pending | approved | edited | rejected
        "trigger_reason": trigger_reason,
        "problem_text":   parsed.get("problem_text", ""),
        "topic":          parsed.get("topic", ""),
        "solution_steps": solution.get("solution", {}).get("steps", []),
        "final_answer":   solution.get("solution", {}).get("final_answer", ""),
        "confidence":     solution.get("confidence", 0.0),
        "verdict":        verification.get("verdict", "unknown"),
        "verifier_issues": (
            verification.get("correctness", {}).get("issues", []) +
            verification.get("units_and_domain", {}).get("issues", []) +
            verification.get("edge_cases", {}).get("issues", [])
        ),
        # Filled by human
        "human_action":        None,   # approve | edit | reject
        "corrected_answer":    None,
        "correction_comment":  None,
        "approved_by":         None,
    }

    _save_record(record)
    return record


# ── Human actions ─────────────────────────────────────────────────────────────

def approve_review(review_id: str, approved_by: str = "reviewer") -> dict:
    """Mark a pending review as approved — solution is correct as-is."""
    return _update_record(review_id, {
        "status":       "approved",
        "human_action": "approve",
        "approved_by":  approved_by,
    })


def edit_review(
    review_id: str,
    corrected_answer: str,
    correction_comment: str,
    approved_by: str = "reviewer",
) -> dict:
    """Human provides a corrected answer — saved as a learning signal."""
    return _update_record(review_id, {
        "status":              "edited",
        "human_action":        "edit",
        "corrected_answer":    corrected_answer,
        "correction_comment":  correction_comment,
        "approved_by":         approved_by,
    })


def reject_review(
    review_id: str,
    correction_comment: str,
    approved_by: str = "reviewer",
) -> dict:
    """Human rejects the solution entirely — flagged for re-solving."""
    return _update_record(review_id, {
        "status":             "rejected",
        "human_action":       "reject",
        "correction_comment": correction_comment,
        "approved_by":        approved_by,
    })


# ── Feedback from student (UI feedback buttons) ───────────────────────────────

def save_student_feedback(
    parsed: dict,
    solution: dict,
    feedback: str,          # "correct" | "incorrect"
    comment: str = "",
) -> dict:
    """
    Save ✅ / ❌ feedback from the student UI.
    This is separate from HITL — it's a lightweight learning signal.
    """
    record = {
        "review_id":       str(uuid.uuid4())[:8],
        "timestamp":       datetime.datetime.utcnow().isoformat(),
        "status":          "student_feedback",
        "trigger_reason":  "student_feedback_button",
        "problem_text":    parsed.get("problem_text", ""),
        "topic":           parsed.get("topic", ""),
        "final_answer":    solution.get("solution", {}).get("final_answer", ""),
        "confidence":      solution.get("confidence", 0.0),
        "feedback":        feedback,          # "correct" | "incorrect"
        "comment":         comment,
        "human_action":    feedback,
        "corrected_answer": None,
        "correction_comment": comment,
        "approved_by":     "student",
    }
    _save_record(record)
    return record


# ── Query helpers ─────────────────────────────────────────────────────────────

def get_pending_reviews() -> list[dict]:
    """Return all records with status=pending."""
    return [r for r in _load_all() if r.get("status") == "pending"]


def get_corrections_for_topic(topic: str) -> list[dict]:
    """
    Return approved edits/corrections for a given topic.
    Used by the memory layer to surface known corrections at solve time.
    """
    return [
        r for r in _load_all()
        if r.get("topic") == topic
        and r.get("human_action") in ("edit", "approve")
        and r.get("corrected_answer") is not None
    ]


def get_all_reviews() -> list[dict]:
    return _load_all()


# ── Persistence helpers ───────────────────────────────────────────────────────

def _save_record(record: dict):
    with open(HITL_STORE, "a") as f:
        f.write(json.dumps(record) + "\n")


def _load_all() -> list[dict]:
    if not HITL_STORE.exists():
        return []
    records = []
    with open(HITL_STORE) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return records


def _update_record(review_id: str, updates: dict) -> dict:
    records = _load_all()
    updated = None
    for r in records:
        if r.get("review_id") == review_id:
            r.update(updates)
            updated = r
            break
    # Rewrite file
    with open(HITL_STORE, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    return updated or {}