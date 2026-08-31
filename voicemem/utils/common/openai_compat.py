"""OpenAI-compatible chat helpers.

LM Studio (and some LiteLLM backends) reject ``response_format={type: json_object}``.
They only accept ``json_schema`` or ``text``. VoiceMem still asks for JSON in the
prompt, so we retry without the OpenAI-only flag and parse the text.
"""
from __future__ import annotations


def chat_complete(client, **kwargs):
    """``client.chat.completions.create`` with a LiteLLM/LMS json_object fallback."""
    try:
        return client.chat.completions.create(**kwargs)
    except Exception as exc:
        err = str(exc).lower()
        if "response_format" not in kwargs:
            raise
        if "response_format" in err or "json_object" in err or "json_schema" in err:
            kwargs = dict(kwargs)
            kwargs.pop("response_format", None)
            return client.chat.completions.create(**kwargs)
        raise


def message_text(resp) -> str:
    msg = resp.choices[0].message
    return ((msg.content or "").strip()
            or (getattr(msg, "reasoning_content", None) or "").strip())
