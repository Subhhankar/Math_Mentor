import json
import re
import math
import traceback
from fractions import Fraction
from langchain_google_genai import ChatGoogleGenerativeAI


# ── Safe Python Calculator Tool ───────────────────────────────────────────────

SAFE_MATH_GLOBALS = {
    "__builtins__": {},
    # math functions
    "sqrt": math.sqrt, "log": math.log, "log2": math.log2, "log10": math.log10,
    "exp": math.exp, "sin": math.sin, "cos": math.cos, "tan": math.tan,
    "asin": math.asin, "acos": math.acos, "atan": math.atan, "atan2": math.atan2,
    "ceil": math.ceil, "floor": math.floor, "abs": abs, "round": round,
    "pow": pow, "factorial": math.factorial, "comb": math.comb, "perm": math.perm,
    "gcd": math.gcd, "lcm": math.lcm,
    # constants
    "pi": math.pi, "e": math.e, "inf": math.inf,
    # fractions
    "Fraction": Fraction,
    # builtins safe for math
    "sum": sum, "min": min, "max": max, "range": range, "list": list,
    "int": int, "float": float, "str": str,
}


def python_calculator(expression: str) -> dict:
    """
    Safely evaluate a Python math expression.

    Args:
        expression: A valid Python math expression string
                    e.g. "comb(6,2) * (1/6)**2 * (5/6)**4"

    Returns:
        {"result": <value>, "expression": <input>, "error": None}
        or
        {"result": None, "expression": <input>, "error": <message>}
    """
    try:
        result = eval(expression, SAFE_MATH_GLOBALS, {})  # noqa: S307
        return {"result": result, "expression": expression, "error": None}
    except Exception:
        return {"result": None, "expression": expression, "error": traceback.format_exc(limit=1)}


# ── Prompts ───────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert mathematics tutor specialized in JEE-level problems (Algebra, Probability, Calculus, Linear Algebra).

You will receive a structured math problem along with retrieved context from a knowledge base.

---

RETRIEVED CONTEXT:
<<CONTEXT>>

---

STRUCTURED PROBLEM:
<<STRUCTURED_PROBLEM>>

---

CONVERSATION HISTORY / SIMILAR SOLVED PROBLEMS:
<<MEMORY_CONTEXT>>

---

CALCULATOR RESULTS (pre-computed for you):
<<CALCULATOR_RESULTS>>

---

Your task is to respond in the following strict JSON format:

{
  "solution": {
    "steps": [
      {
        "step_number": 1,
        "description": "Brief label for this step",
        "work": "Detailed mathematical working",
        "reasoning": "Why this step is taken"
      }
    ],
    "final_answer": "The final answer with units/domain if applicable",
    "answer_latex": "LaTeX representation of the final answer"
  },
  "explanation": "A clear, student-friendly explanation of the overall approach in 2-4 sentences",
  "topic": "The math topic this problem belongs to",
  "formulas_used": ["List of formulas or identities applied"],
  "confidence": 0.0,
  "confidence_reason": "Brief reason for your confidence score",
  "needs_human_review": false,
  "review_reason": "",
  "edge_cases": "Any domain restrictions, special cases, or caveats",
  "sources_used": ["Titles or labels of retrieved context chunks that were actually used"]
}

---

RULES:
- confidence must be a float between 0.0 and 1.0
- If confidence < 0.75, set needs_human_review to true and add a review_reason
- If CALCULATOR RESULTS are provided, use them directly in your steps — do not recompute
- Do NOT hallucinate formulas or cite sources not present in retrieved context
- If retrieved context is insufficient, say so in confidence_reason
- Steps must be complete enough for a student to follow independently
- Never skip steps; show all algebraic manipulations
- IMPORTANT: Solve the exact problem provided. Do not solve a generic or example problem.
- Keep each step's "work" field concise — max 3-4 lines. Do not over-explain.
- Keep "reasoning" to 1-2 sentences per step.
- CRITICAL: Show FULL mathematical working in each step. Do NOT just state the answer like "a = 2". 
  Show every substitution, algebraic manipulation, and calculation explicitly.
- CRITICAL FORMATTING: Write math in plain text exactly like handwriting in a notebook.
  Do NOT use LaTeX, do NOT use dollar signs ($), do NOT use markdown.
  Use plain unicode symbols: x * y or 2x, x^2 for powers, sqrt(x) for roots
  Use fractions like: b/x^2 not frac notation
  Example of GOOD work:
    g'(x) = 2ax - b/x^2
    g'(4) = 0
    2a(4) - b/16 = 0
    8a = b/16
    b = 128a  ... (Equation 1)
"""

# Prompt asking Gemini to extract calculator expressions from the problem
CALC_EXTRACTION_PROMPT = """You are a math assistant. Given the problem below, extract Python math expressions
that should be pre-computed by a calculator to help solve it.

Use only: sqrt, log, log10, exp, sin, cos, tan, pi, e, factorial, comb, perm, gcd, Fraction, abs, round, pow

Return ONLY a JSON array of expression strings. If no calculation is needed, return [].
No markdown, no explanation.

Example: ["comb(6,2)", "Fraction(1,6) + Fraction(1,4)", "factorial(5) / factorial(3)"]

Problem:
<<PROBLEM_TEXT>>
"""


class SolverAgent:
    """
    Solver Agent: Uses retriever + Gemini + optional Python calculator tool
    to solve structured math problems via RAG.
    Calls Gemini directly to ensure all prompt variables are injected correctly.
    """

    def __init__(self, retriever):
        self.retriever = retriever
        self.chat_model = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0.3,
            max_tokens=16000,
        )

    def solve(self, routing: dict, memory_context: str = "") -> dict:
        """
        Solve the problem using RAG + optional calculator tool.

        Args:
            routing: Output dict from IntentRouterAgent
            memory_context: Similar solved problems from memory layer (pass "" if none)

        Returns:
            Solution dict with steps, answer, confidence, HITL flag, and calculator trace
        """
        problem_text = routing.get("problem_text", "")
        if not problem_text:
            return self._error_result("No problem text found in routing dict.")

        structured_summary = (
            f"Topic: {routing.get('topic', 'unknown')}\n"
            f"Sub-type: {routing.get('sub_type', 'unknown')}\n"
            f"Variables: {routing.get('variables', [])}\n"
            f"Constraints: {routing.get('constraints', [])}"
        )

        # Step 1: Run calculator if routing requests it
        calculator_results = "Not used."
        calculator_trace = []
        if routing.get("requires_calculator", False):
            calculator_trace = self._run_calculator(problem_text)
            if calculator_trace:
                calculator_results = "\n".join(
                    f"  {r['expression']} = {r['result']}"
                    if r["error"] is None
                    else f"  {r['expression']} → ERROR: {r['error']}"
                    for r in calculator_trace
                )

        # Step 2: Retrieve relevant docs from vector store
        try:
            retrieved_docs = self.retriever.invoke(problem_text)
            context_text = "\n\n".join(
                f"[Chunk {i+1}]\n{doc.page_content}"
                for i, doc in enumerate(retrieved_docs)
            ) if retrieved_docs else "No relevant context found."
        except Exception as retr_err:
            retrieved_docs = []
            context_text = f"Retrieval failed ({str(retr_err)}) — solving from Gemini knowledge only."

        # Step 3: Build prompt using safe string replacement (NOT .format())
        # .format() breaks if SYSTEM_PROMPT contains any {word} like {b}, {x}, etc.
        try:
            full_prompt = SYSTEM_PROMPT \
                .replace("<<CONTEXT>>", context_text) \
                .replace("<<STRUCTURED_PROBLEM>>", structured_summary) \
                .replace("<<MEMORY_CONTEXT>>", memory_context or "No similar problems found.") \
                .replace("<<CALCULATOR_RESULTS>>", calculator_results)
        except Exception as fmt_err:
            return self._error_result(f"Prompt formatting failed: {str(fmt_err)}")

        # Step 4: Call Gemini directly with the complete prompt + actual problem
        try:
            from langchain_core.messages import HumanMessage, SystemMessage
            response = self.chat_model.invoke([
                SystemMessage(content=full_prompt),
                HumanMessage(content=f"Solve this problem:\n\n{problem_text}"),
            ])

            raw_answer = response.content

            # DEBUG — write raw response to file
            with open("solver_debug.txt", "w", encoding="utf-8") as dbg:
                dbg.write(raw_answer)

            result = self._extract_json(raw_answer)
            result["retrieved_docs"] = [
                doc.page_content[:200] for doc in retrieved_docs
            ]
            result["calculator_trace"] = calculator_trace
            result["rag_used"] = bool(retrieved_docs)
            return result

        except Exception as e:
            with open("solver_debug.txt", "a", encoding="utf-8") as dbg:
                dbg.write(f"\n\nERROR: {str(e)}")
            return self._error_result(f"Solver failed: {str(e)}")

    def _run_calculator(self, problem_text: str) -> list:
        """Ask Gemini to extract expressions, then evaluate them safely."""
        try:
            extraction_prompt = CALC_EXTRACTION_PROMPT.replace("<<PROBLEM_TEXT>>", problem_text)
            response = self.chat_model.invoke(extraction_prompt)
            raw = response.content.strip()
            raw = re.sub(r"^```(?:json)?\s*", "", raw)
            raw = re.sub(r"\s*```$", "", raw)
            expressions = json.loads(raw)

            if not isinstance(expressions, list):
                return []

            return [python_calculator(expr) for expr in expressions]

        except Exception:
            return []

    def _extract_json(self, text: str) -> dict:
        text = text.strip()
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        text = text.strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        # Find outermost { }
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(text[start:end+1])
        raise ValueError(f"No JSON found in response: {text[:300]}")

    def _error_result(self, reason: str) -> dict:
        return {
            "solution": {"steps": [], "final_answer": "", "answer_latex": ""},
            "explanation": "",
            "topic": "unknown",
            "formulas_used": [],
            "confidence": 0.0,
            "confidence_reason": reason,
            "needs_human_review": True,
            "review_reason": reason,
            "edge_cases": "",
            "sources_used": [],
            "retrieved_docs": [],
        }


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    load_dotenv()

    from vector_store import get_retriever
    retriever = get_retriever(k=3)

    agent = SolverAgent(retriever=retriever)

    routing = {
        "problem_text": (
            "One die has two faces marked 1, two faces marked 2, one face marked 3 "
            "and one face marked 4. Another die has one face marked 1, two faces marked 2, "
            "two faces marked 3 and one face marked 4. Find the probability of getting "
            "sum 4 or 5 when both dice are thrown."
        ),
        "topic": "probability",
        "sub_type": "dice probability",
        "variables": [],
        "constraints": [],
    }

    result = agent.solve(routing, memory_context="")
    print(json.dumps(result, indent=2))