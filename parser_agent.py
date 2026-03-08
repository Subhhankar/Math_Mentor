import json
import re
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv
load_dotenv()


class ParserAgent:
    """
    Parser Agent: Cleans OCR/ASR/text input and converts it
    into a structured math problem format.
    """

    def __init__(self, gemini_api_key: str):
        self.model = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0,
            max_tokens=1024,
            google_api_key=gemini_api_key,
        )

        self.system_prompt = """You are a math problem parser for a JEE-style tutoring system.
Your job is to:
1. Clean noisy OCR or speech-to-text input
2. Identify the math topic
3. Extract variables and constraints
4. Structure the problem clearly

IMPORTANT RULES:
- Be LENIENT. Almost any math input is a valid problem.
- If the student writes a formula or identity, assume they want to understand, prove, or apply it.
- If the student writes an equation, assume they want to solve it.
- If the student writes an expression, assume they want to simplify or evaluate it.
- Only set needs_clarification=true if the input is completely empty or total gibberish with no math at all.
- NEVER reject a problem just because it lacks an explicit question word like "find" or "prove".
- When in doubt, set needs_clarification=false and do your best to interpret the problem.

Always respond with ONLY a valid JSON object. No explanation, no markdown.

JSON format:
{
  "problem_text": "<cleaned, well-formatted problem statement>",
  "topic": "<one of: algebra | probability | calculus | linear_algebra | arithmetic | other>",
  "variables": ["<list of variable names found>"],
  "constraints": ["<list of constraints or conditions mentioned>"],
  "needs_clarification": false,
  "clarification_reason": ""
}"""

    def parse(self, raw_input: str, input_source: str = "text") -> dict:
        if not raw_input or not raw_input.strip():
            return self._error_result("Empty input received.")

        source_hints = {
            "ocr": "This text was extracted via OCR and may contain noise.",
            "asr": "This text came from speech recognition — convert spoken math to notation.",
            "text": "This text was typed directly by the user.",
        }
        hint = source_hints.get(input_source, "")

        prompt = f"""{self.system_prompt}

Input source: {input_source}
Note: {hint}

Raw input:
\"\"\"{raw_input}\"\"\"

Respond with the JSON object only."""

        try:
            response = self.model.invoke([
                SystemMessage(content=prompt),
                HumanMessage(content=raw_input),
            ])
            result = self._extract_json(response.content)
            result["input_source"] = input_source
            result["raw_input"] = raw_input
            # Extra safety: never block on clarification unless truly empty
            if not raw_input.strip():
                result["needs_clarification"] = True
            else:
                result["needs_clarification"] = False
                result["clarification_reason"] = ""
            return result

        except Exception as e:
            return self._error_result(f"Parser failed: {str(e)}", raw_input)

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
        raise ValueError(f"Could not parse parser JSON: {text[:200]}")

    def _error_result(self, reason: str, raw_input: str = "") -> dict:
        return {
            "problem_text": raw_input,
            "topic": "other",
            "variables": [],
            "constraints": [],
            "needs_clarification": False,
            "clarification_reason": "",
            "input_source": "unknown",
            "raw_input": raw_input,
        }