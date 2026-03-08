import json
import re
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage


SYSTEM_PROMPT = """You are a strict mathematical verifier for a JEE-level tutoring system.

Verify the solution across three dimensions:
1. CORRECTNESS — Are the steps logically valid? Is the final answer correct?
2. UNITS & DOMAIN — Are units consistent? Is the answer in valid domain (e.g. probability in [0,1])?
3. EDGE CASES — Were special cases handled?

---

ORIGINAL PROBLEM:
<<PROBLEM_TEXT>>

---

SOLUTION TO VERIFY:
<<SOLUTION>>

---

CALCULATOR VERIFICATION RESULTS:
<<CALCULATOR_CHECK>>

---

Respond ONLY with a valid JSON object. No markdown, no explanation, no extra text.

{
  "verdict": "<pass | fail | uncertain>",
  "correctness": {
    "status": "<pass | fail | uncertain>",
    "issues": ["<list any correctness issues, empty list if none>"]
  },
  "units_and_domain": {
    "status": "<pass | fail | uncertain>",
    "issues": ["<list any unit or domain violations, empty list if none>"]
  },
  "edge_cases": {
    "status": "<pass | fail | uncertain>",
    "issues": ["<list any unhandled edge cases, empty list if none>"]
  },
  "confidence": 0.0,
  "confidence_reason": "<brief reason for confidence score>",
  "needs_hitl": false,
  "hitl_reason": "<reason if needs_hitl is true, else empty string>",
  "suggested_fix": "<brief fix suggestion if verdict is fail or uncertain, else empty string>"
}

RULES:
- verdict is "pass" only if ALL three checks pass
- verdict is "fail" if any check clearly fails
- verdict is "uncertain" if you cannot fully verify
- needs_hitl must be true if verdict is "fail" or "uncertain" OR confidence < 0.75
- Keep all text fields brief — 1-2 sentences maximum
- issues lists should have short entries, not long explanations
"""


class VerifierAgent:
    """
    Verifier / Critic Agent: Checks correctness, units & domain,
    and edge cases of the Solver's solution.
    """

    def __init__(self, gemini_api_key: str = None):
        kwargs = {"google_api_key": gemini_api_key} if gemini_api_key else {}
        self.model = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0.1,
            max_tokens=4096,
            **kwargs,
        )

    def verify(self, routing: dict, solution: dict) -> dict:
        problem_text = routing.get("problem_text", "")
        if not problem_text:
            return self._error_result("No problem text provided for verification.")

        # Check if solver already failed
        if not solution.get("solution", {}).get("final_answer"):
            return self._error_result(
                "The solution is entirely missing, making any form of verification impossible. "
                "Human intervention is required to obtain a solution."
            )

        solution_summary = self._format_solution(solution)
        calculator_check = self._recompute(solution)

        try:
            full_prompt = SYSTEM_PROMPT \
                .replace("<<PROBLEM_TEXT>>", problem_text) \
                .replace("<<SOLUTION>>", solution_summary) \
                .replace("<<CALCULATOR_CHECK>>", calculator_check)
        except Exception as fmt_err:
            return self._error_result(f"Prompt formatting failed: {str(fmt_err)}")

        try:
            response = self.model.invoke([
                SystemMessage(content=full_prompt),
                HumanMessage(content="Please verify the solution above. Respond with JSON only."),
            ])

            result = self._extract_json(response.content)

            # Only force HITL if solver itself explicitly flagged review needed
            if solution.get("needs_human_review") and not result.get("needs_hitl"):
                result["needs_hitl"] = True
                result["hitl_reason"] = (
                    "Solver flagged low confidence: "
                    + solution.get("review_reason", "")
                )

            result["problem_text"] = problem_text
            return result

        except Exception as e:
            return self._error_result(f"Verifier failed: {str(e)}")

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _format_solution(self, solution: dict) -> str:
        sol = solution.get("solution", {})
        steps = sol.get("steps", [])
        steps_text = "\n".join(
            f"  Step {s.get('step_number')}: {s.get('description')} | {s.get('work', '')[:150]}"
            for s in steps
        )
        return (
            f"Steps:\n{steps_text}\n\n"
            f"Final Answer: {sol.get('final_answer', '')}\n"
            f"Formulas Used: {solution.get('formulas_used', [])}\n"
            f"Solver Confidence: {solution.get('confidence', 'N/A')}"
        )

    def _recompute(self, solution: dict) -> str:
        traces = solution.get("calculator_trace", [])
        if not traces:
            return "No calculator results available."
        lines = []
        for t in traces:
            if t.get("error") is None:
                lines.append(f"  {t['expression']} = {t['result']}  ✓")
            else:
                lines.append(f"  {t['expression']} → ERROR: {t['error']}")
        return "\n".join(lines)

    def _extract_json(self, text: str) -> dict:
        text = text.strip()
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        text = text.strip()

        # Try direct parse first
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Find outermost { }
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(text[start:end+1])
            except json.JSONDecodeError:
                pass

        raise ValueError(f"Could not parse JSON from verifier response: {text[:300]}")

    def _error_result(self, reason: str) -> dict:
        return {
            "verdict": "uncertain",
            "correctness": {"status": "uncertain", "issues": [reason]},
            "units_and_domain": {"status": "uncertain", "issues": []},
            "edge_cases": {"status": "uncertain", "issues": []},
            "confidence": 0.0,
            "confidence_reason": reason,
            "needs_hitl": True,
            "hitl_reason": reason,
            "suggested_fix": "",
            "problem_text": "",
        }