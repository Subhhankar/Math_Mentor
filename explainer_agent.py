import json
import re
from langchain_google_genai import ChatGoogleGenerativeAI


SYSTEM_PROMPT = """You are a friendly, patient JEE math tutor explaining a solved problem to a student.

You will receive the original problem, its solution, and the verification report.
Your job is to produce a clear, student-friendly explanation that helps the student truly understand — not just see the answer.

---

ORIGINAL PROBLEM:
<<PROBLEM_TEXT>>

---

VERIFIED SOLUTION:
<<SOLUTION>>

---

VERIFICATION REPORT:
<<VERIFICATION>>

---

Respond ONLY with a valid JSON object. No markdown, no explanation outside the JSON.

{
  "title": "Short descriptive title for this problem",
  "concept_intro": "1-2 sentences introducing the core concept needed (e.g. what is conditional probability)",
  "steps": [
    {
      "step_number": 1,
      "heading": "Short heading for this step",
      "explanation": "Plain English explanation of what we are doing and why",
      "math": "The actual mathematical working for this step",
      "tip": "Optional tip or common mistake to avoid at this step (empty string if none)"
    }
  ],
  "final_answer": "The final answer stated clearly",
  "summary": "2-3 sentence recap of the approach used",
  "key_concepts": ["List of concepts a student should know to solve this type of problem"],
  "common_mistakes": ["List of mistakes students commonly make on this problem type"],
  "difficulty": "<easy | medium | hard>",
  "topic": "Math topic",
  "follow_up_problems": ["1-2 similar problems a student can try for practice"]
}

RULES:
- Use simple language — write as if talking to a 17-year-old student
- Every step must have a plain English explanation, not just math
- If the verifier found issues, acknowledge them honestly in the summary
- Do NOT fabricate steps not present in the solution
- keep follow_up_problems relevant to the same topic and difficulty
"""


class ExplainerAgent:
    """
    Explainer / Tutor Agent: Converts a verified solution into a
    student-friendly step-by-step explanation.
    """

    def __init__(self, gemini_api_key: str = None):
        kwargs = {"google_api_key": gemini_api_key} if gemini_api_key else {}
        self.model = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0.4,
            max_tokens=4096,
            **kwargs,
        )

    def explain(self, routing: dict, solution: dict, verification: dict) -> dict:
        """
        Generate a student-friendly explanation.

        Args:
            routing:      Output from IntentRouterAgent (contains problem_text)
            solution:     Output from SolverAgent
            verification: Output from VerifierAgent

        Returns:
            Explanation dict with steps, concepts, tips, and follow-up problems
        """
        problem_text = routing.get("problem_text", "")
        if not problem_text:
            return self._error_result("No problem text found.")

        from langchain_core.messages import HumanMessage, SystemMessage

        full_prompt = SYSTEM_PROMPT \
            .replace("<<PROBLEM_TEXT>>", problem_text) \
            .replace("<<SOLUTION>>", self._format_solution(solution)) \
            .replace("<<VERIFICATION>>", self._format_verification(verification))

        try:
            response = self.model.invoke([
                SystemMessage(content=full_prompt),
                HumanMessage(content="Please explain this solution to the student."),
            ])

            result = self._extract_json(response.content)
            result["needs_hitl"] = False
            return result

        except Exception as e:
            return self._error_result(f"Explainer failed: {str(e)}")

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _format_solution(self, solution: dict) -> str:
        sol = solution.get("solution", {})
        steps = sol.get("steps", [])
        steps_text = "\n".join(
            f"  Step {s.get('step_number')} — {s.get('description')}:\n"
            f"    {s.get('work')}"
            for s in steps
        )
        return (
            f"Steps:\n{steps_text}\n\n"
            f"Final Answer: {sol.get('final_answer', '')}\n"
            f"Formulas Used: {solution.get('formulas_used', [])}\n"
            f"Edge Cases: {solution.get('edge_cases', 'None noted')}"
        )

    def _format_verification(self, verification: dict) -> str:
        return (
            f"Verdict: {verification.get('verdict', 'unknown')}\n"
            f"Correctness: {verification.get('correctness', {}).get('status')} "
            f"— {verification.get('correctness', {}).get('issues', [])}\n"
            f"Domain/Units: {verification.get('units_and_domain', {}).get('status')} "
            f"— {verification.get('units_and_domain', {}).get('issues', [])}\n"
            f"Edge Cases: {verification.get('edge_cases', {}).get('status')} "
            f"— {verification.get('edge_cases', {}).get('issues', [])}\n"
            f"Suggested Fix: {verification.get('suggested_fix', 'None')}"
        )

    def _extract_json(self, text: str) -> dict:
        text = text.strip()
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        text = text.strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(text[start:end+1])
            except json.JSONDecodeError:
                pass
        raise ValueError(f"Could not parse explainer JSON: {text[:300]}")

    def _error_result(self, reason: str) -> dict:
        return {
            "title": "",
            "concept_intro": reason,
            "steps": [],
            "final_answer": "",
            "summary": "",
            "key_concepts": [],
            "common_mistakes": [],
            "difficulty": "unknown",
            "topic": "unknown",
            "follow_up_problems": [],
            "needs_hitl": False,
        }