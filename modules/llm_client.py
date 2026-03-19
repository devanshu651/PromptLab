"""
modules/llm_client.py  — v3
─────────────────────────────
OpenRouter client with latency + token tracking.
"""
from __future__ import annotations
import os, time
from openai import OpenAI
import streamlit as st

APP_VERSION = "3.0.0"


def get_client() -> OpenAI:
    api_key = os.getenv("OPENROUTER_API_KEY") or st.secrets.get("OPENROUTER_API_KEY", "")
    if not api_key:
        st.error("🔑 **OPENROUTER_API_KEY** not found. Add to .env or Streamlit Secrets.")
        st.stop()
    return OpenAI(
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
        default_headers={
            "HTTP-Referer": "https://promptlab-dev.streamlit.app",
            "X-Title": f"PromptLab Pro v{APP_VERSION}",
        },
    )


def call_llm(
    client: OpenAI,
    prompt: str,
    model: str,
    temperature: float,
    max_tokens: int = 512,
    json_mode: bool = False,
) -> dict:
    t0 = time.time()
    try:
        kwargs: dict = dict(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        resp  = client.chat.completions.create(**kwargs)
        text  = resp.choices[0].message.content.strip()
        usage = resp.usage
        return {
            "text"              : text,
            "tokens_prompt"     : getattr(usage, "prompt_tokens", 0),
            "tokens_completion" : getattr(usage, "completion_tokens", 0),
            "latency_ms"        : int((time.time() - t0) * 1000),
            "error"             : None,
        }
    except Exception as e:
        return {
            "text": "", "tokens_prompt": 0, "tokens_completion": 0,
            "latency_ms": int((time.time() - t0) * 1000), "error": str(e),
        }