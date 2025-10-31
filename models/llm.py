"""LLM wrapper with safe fallback.

If `transformers` is installed we use a text-generation pipeline (small model).
Otherwise we provide a deterministic fallback echo/suggestion generator.
"""
from typing import Optional


class LLM:
    def __init__(self):
        self._use_transformers = False
        try:
            from transformers import pipeline  # type: ignore
            self._use_transformers = True
            self._pipe = pipeline("text-generation", model="distilgpt2", device=-1)
        except Exception:
            self._use_transformers = False
            self._pipe = None

    def generate(self, prompt: str, max_length: int = 64) -> str:
        if self._use_transformers and self._pipe is not None:
            out = self._pipe(prompt, max_length=max_length, do_sample=True, num_return_sequences=1)
            return out[0]["generated_text"]
        # fallback: simple deterministic reply
        if "recommend" in prompt.lower() or "suggest" in prompt.lower():
            return "Try asking for a genre, mood, or mention a movie you like. For example: 'recommend sci-fi movies'."
        if "like" in prompt.lower():
            return "Tell me the movie title in full (e.g. 'I like Inception') and I'll find similar films."
        return "I can recommend movies or find similar titles. Ask for 'recommend' or say 'I like <movie>'."


_default_llm: Optional[LLM] = None


def get_llm() -> LLM:
    global _default_llm
    if _default_llm is None:
        _default_llm = LLM()
    return _default_llm
