"""LLM-as-judge: scores whether an answer is actually faithful to the sources it was given,
not just whether it looks plausible. A second, independent model call which reads the answer
and the sources together and judges faithfulness.
"""

from __future__ import annotations
import json
import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(override=True)

_client = None
MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
BASE_URL = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
API_KEY = os.getenv("GROQ_API_KEY", "")

JUDGE_PROMPT = """You are a strict fact-checker. Read the ANSWER and the SOURCES it was supposedly
grounded in. Judge whether every factual claim in the answer is actually supported by the sources -
not whether the answer is well-written, not whether it's plausible, only whether it's TRUE according
to the sources given.

SOURCES:
{sources}

ANSWER:
{answer}

Respond with only a JSON object, no other text:
{{"faithful": true or false, "reason": "one sentence explaining your judgement"}}

Mark faithful = false if the answer states anything not supported by the sources, evem a small detail,
or it it hedges by citing a source that doesn't actually say the claim.
"""

def _get_client():
    global _client
    if _client is None:
        _client = OpenAI(
            base_url=BASE_URL,
            api_key=API_KEY
        )
    return _client

def judge_faithfulness(answer, sources_text):
    """Ask a second model call whether the `answer` is actually suppported by the `sources_text`.
    Returns {faithful: bool, reason: str}

    Deliberately separate call from the one that generated the answer
    """

    prompt = JUDGE_PROMPT.format(sources=sources_text, answer=answer)
    response = _get_client().chat.completions.create(
        model = MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    raw = response.choices[0].message.content.strip()
    if raw.startswith("```"):
        raw = raw.strip("`").removeprefix("json").strip()
    try:
        result = json.loads(raw)
        return bool(result["faithful"]), result["reason"]
    except:
        print("Judge returned non-JSON, defaulting to non-faithful")
        return False, "Failed to parse judge output."
