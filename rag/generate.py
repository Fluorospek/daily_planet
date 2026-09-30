from __future__ import annotations
import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(override=True)

_client = None
MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
BASE_URL = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
API_KEY = os.getenv("GROQ_API_KEY", "")

def _get_client():
    global _client
    if _client is None:
        _client = OpenAI(
            base_url=BASE_URL,
            api_key=API_KEY
        )
    return _client

def format_sources(chunks):
    lines = []
    for i, chunk in enumerate(chunks, start=1):
        meta = chunk["metadata"]
        lines.append(f"[Source {i}: {meta.get('source')}]\n{chunk['text']}")
    return "\n\n".join(lines)

NAIVE_PROMPT = """Answer the question using the context provided if it's helpful.

{context}

Question: {question}
Answer:"""

GROUNDED_PROMPT="""Answer the qquestion using ONLY the sources provided below. Cite the source number\
    after every claim, like [1]. If the source does not contain the answer, say exactly: \
    "I can't verify that from the Daily Planet's sources." Do not use any outside knowledge, even if you are \
    confident that it's correct

    {context}

    Question: {question}
    Answer: """

def generate_answer(question, chunks, prompt_template=GROUNDED_PROMPT):
    """Turn the retrieved chunks + a question into a answer, using the given prompt"""
    context = format_sources(chunks)
    prompt = prompt_template.format(context = context, question = question)

    response = _get_client().chat.completions.create(
        model = MODEL,
        messages=[{'role': 'user', "content": prompt}],
        temperature = 0
    )
    return response.choices[0].message.content