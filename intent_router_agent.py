import json
import re
import os
import google.generativeai as genai

from dotenv import load_dotenv
load_dotenv()
class IntentRouterAgent:
    """
    Intent Router Agent: Receives structured problem from ParserAgent,
    classifies intent, selects solving strategy, and returns a routing plan.
    """

    TOPICS = ["algebra", "probability", "calculus", "linear_algebra", "arithmetic", "other"]

    STRATEGIES = {
        "algebra":       ["symbolic_solver", "rag_retrieval", "python_calculator"],
        "probability":   ["rag_retrieval", "python_calculator"],
        "calculus":      ["symbolic_solver", "rag_retrieval"],
        "linear_algebra":["symbolic_solver", "rag_retrieval", "python_calculator"],
        "arithmetic":    ["python_calculator"],
        "other":         ["rag_retrieval"],
    }

    def __init__(self, GOOGLE_API_KEY: str):
        genai.configure(api_key=GOOGLE_API_KEY)
        self.model = genai.GenerativeModel("gemini-2.5-flash")

        self.system_prompt = """You are an intent router for a JEE-style math tutoring system.

Given a structured math problem, you must:
1. Confirm or refine the topic classification
2. Identify the problem sub-type (e.g. quadratic, integration, bayes theorem)
3. Choose the best solving strategy
4. Set priority order of tools/agents to invoke
5. Flag if the problem is out of scope

Allowed topics: algebra | probability | calculus | linear_algebra | arithmetic | other
Allowed strategies: symbolic_solver | rag_retrieval | python_calculator | web_search

Respond ONLY with a valid JSON object. No markdown, no explanation.

JSON format:
{
  "topic": "<confirmed topic>",
  "sub_type": "<specific problem type, e.g. 'quadratic equation', 'bayes theorem', 'definite integral'>",
  "intent": "<solve | simplify | verify | explain | out_of_scope>",
  "strategy": ["<ordered list of strategies to apply>"],
  "requires_calculator": <true | false>,
  "requires_rag": <true | false>,
  "out_of_scope": <true | false>,
  "out_of_scope_reason": "<reason if out_of_scope else empty string>",
  "routing_notes": "<brief note on why this routing was chosen>"
}"""

    def route(self, parsed_problem: dict) -> dict:
        """
        Route a parsed problem to the correct solving pipeline.

        Args:
            parsed_problem: Output dict from ParserAgent

        Returns:
            Routing plan dict
        """
        prompt = self._build_prompt(parsed_problem)

        try:
            response = self.model.generate_content(prompt)
            result = self._extract_json(response.text)
            result["problem_text"] = parsed_problem.get("problem_text", "")
            result["variables"] = parsed_problem.get("variables", [])
            result["constraints"] = parsed_problem.get("constraints", [])
            return result

        except Exception as e:
            return self._error_result(f"Router failed: {str(e)}", parsed_problem)

    def _build_prompt(self, parsed_problem: dict) -> str:
        return f"""{self.system_prompt}

Structured problem:
{json.dumps(parsed_problem, indent=2)}

Respond with the JSON object only."""

    def _extract_json(self, text: str) -> dict:
        text = text.strip()
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        return json.loads(text)

    def _hitl_result(self, reason: str) -> dict:
        return {
            "topic": "unknown",
            "sub_type": "unknown",
            "intent": "clarify",
            "strategy": [],
            "requires_calculator": False,
            "requires_rag": False,
            "out_of_scope": False,
            "out_of_scope_reason": "",
            "routing_notes": reason,
            "needs_hitl": True,
        }

    def _error_result(self, reason: str, parsed_problem: dict) -> dict:
        topic = parsed_problem.get("topic", "other")
        fallback_strategy = self.STRATEGIES.get(topic, ["rag_retrieval"])
        return {
            "topic": topic,
            "sub_type": "unknown",
            "intent": "solve",
            "strategy": fallback_strategy,
            "requires_calculator": False,
            "requires_rag": True,
            "out_of_scope": False,
            "out_of_scope_reason": "",
            "routing_notes": f"Fallback routing used. Reason: {reason}",
            "needs_hitl": False,
        }


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import os

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("Set GEMINI_API_KEY environment variable.")

    router = IntentRouterAgent(GOOGLE_API_KEY=api_key)

    test_cases = [
        {
            "problem_text": "If X and Y are independent events with P(X)=0.3 and P(Y)=0.5, find P(X∪Y)",
            "topic": "probability",
            "variables": ["X", "Y"],
            "constraints": ["P(X)=0.3", "P(Y)=0.5"],
            "needs_clarification": False,
            "clarification_reason": "",
        },
        {
            "problem_text": "Find the derivative of f(x) = x³ - 4x² + 7",
            "topic": "calculus",
            "variables": ["x"],
            "constraints": [],
            "needs_clarification": False,
            "clarification_reason": "",
        },
        {
            "problem_text": "Who won the cricket world cup?",
            "topic": "other",
            "variables": [],
            "constraints": [],
            "needs_clarification": False,
            "clarification_reason": "",
        },
    ]

    for tc in test_cases:
        print(f"\n{'='*50}")
        print(f"Problem: {tc['problem_text']}")
        result = router.route(tc)
        print("Routing:", json.dumps(result, indent=2))