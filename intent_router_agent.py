import json
import re
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage


class IntentRouterAgent:
    TOPICS = ["algebra", "probability", "calculus", "linear_algebra", "arithmetic", "other"]

    STRATEGIES = {
        "algebra":        ["symbolic_solver", "rag_retrieval", "python_calculator"],
        "probability":    ["rag_retrieval", "python_calculator"],
        "calculus":       ["symbolic_solver", "rag_retrieval"],
        "linear_algebra": ["symbolic_solver", "rag_retrieval", "python_calculator"],
        "arithmetic":     ["python_calculator"],
        "other":          ["rag_retrieval"],
    }

    SYSTEM_PROMPT = """You are an intent router for a JEE-style math tutoring system.

Given a structured math problem, you must:
1. Confirm or refine the topic classification
2. Identify the problem sub-type (e.g. quadratic, integration, bayes theorem)
3. Choose the best solving strategy
4. Set priority order of tools/agents to invoke
5. Flag if the problem is out of scope

Allowed topics: algebra | probability | calculus | linear_algebra | arithmetic | other
Allowed strategies: symbolic_solver | rag_retrieval | python_calculator

Respond ONLY with a valid JSON object. No markdown, no explanation.

{
  "topic": "<confirmed topic>",
  "sub_type": "<specific problem type>",
  "intent": "<solve | simplify | verify | explain | out_of_scope>",
  "strategy": ["<ordered list of strategies>"],
  "requires_calculator": false,
  "requires_rag": true,
  "out_of_scope": false,
  "out_of_scope_reason": "",
  "routing_notes": "<brief note on routing choice>"
}"""

    def __init__(self, gemini_api_key: str):
        self.model = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0.1,
            max_tokens=1024,
            google_api_key=gemini_api_key,
        )

    def route(self, parsed_problem: dict) -> dict:
        if parsed_problem.get("needs_clarification"):
            return self._hitl_result("Parser flagged ambiguity.")

        prompt = f"Structured problem:\n{json.dumps(parsed_problem, indent=2)}\n\nRespond with JSON only."

        try:
            response = self.model.invoke([
                SystemMessage(content=self.SYSTEM_PROMPT),
                HumanMessage(content=prompt),
            ])
            result = self._extract_json(response.content)
            result["problem_text"]  = parsed_problem.get("problem_text", "")
            result["variables"]     = parsed_problem.get("variables", [])
            result["constraints"]   = parsed_problem.get("constraints", [])
            return result

        except Exception as e:
            return self._error_result(f"Router failed: {str(e)}", parsed_problem)

    def _extract_json(self, text: str) -> dict:
        text = text.strip()
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if match:
                return json.loads(match.group())
            raise

    def _hitl_result(self, reason: str) -> dict:
        return {
            "topic": "unknown", "sub_type": "unknown", "intent": "clarify",
            "strategy": [], "requires_calculator": False, "requires_rag": False,
            "out_of_scope": False, "out_of_scope_reason": "",
            "routing_notes": reason, "needs_hitl": True,
        }

    def _error_result(self, reason: str, parsed_problem: dict) -> dict:
        topic = parsed_problem.get("topic", "other")
        return {
            "topic": topic, "sub_type": "unknown", "intent": "solve",
            "strategy": self.STRATEGIES.get(topic, ["rag_retrieval"]),
            "requires_calculator": False, "requires_rag": True,
            "out_of_scope": False, "out_of_scope_reason": "",
            "routing_notes": f"Fallback routing. Reason: {reason}",
            "needs_hitl": False,
            "problem_text": parsed_problem.get("problem_text", ""),
            "variables": parsed_problem.get("variables", []),
            "constraints": parsed_problem.get("constraints", []),
        }
