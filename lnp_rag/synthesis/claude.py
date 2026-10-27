"""Claude provider for evidence synthesis and (optional) query paraphrasing."""
from __future__ import annotations

import json
import logging

import anthropic

from lnp_rag.config import Settings
from lnp_rag.synthesis.prompts import (
    PARAPHRASE_SCHEMA,
    PARAPHRASE_SYSTEM,
    SYNTHESIS_SCHEMA,
    SYNTHESIS_SYSTEM,
    SynthesisError,
    paraphrase_user_message,
    synthesis_user_message,
)

log = logging.getLogger(__name__)


class ClaudeClient:
    name = "claude"

    def __init__(self, settings: Settings):
        kwargs: dict = {"max_retries": settings.claude_max_retries, "timeout": settings.claude_timeout_s}
        # Without an explicit key the SDK resolves ANTHROPIC_API_KEY / auth profiles itself.
        if key := settings.anthropic_key():
            kwargs["api_key"] = key
        self._client = anthropic.Anthropic(**kwargs)
        self.model = settings.claude_model
        self.effort = settings.claude_effort
        self.fallbacks = settings.claude_fallbacks.strip()

    def _call(self, system: str, user: str, schema: dict, max_tokens: int, effort: str) -> dict:
        kwargs: dict = {}
        if self.fallbacks:
            # Server-side fallback: if a safety classifier declines, the API retries
            # on Anthropic's recommended fallback model within the same call.
            kwargs["betas"] = ["server-side-fallback-2026-07-01"]
            kwargs["fallbacks"] = self.fallbacks
        try:
            response = self._client.beta.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                system=system,
                messages=[{"role": "user", "content": user}],
                output_config={"effort": effort, "format": {"type": "json_schema", "schema": schema}},
                **kwargs,
            )
        except anthropic.AuthenticationError as exc:
            raise SynthesisError("Claude authentication failed - check ANTHROPIC_API_KEY") from exc
        except anthropic.BadRequestError as exc:
            raise SynthesisError(f"Claude rejected the request: {exc.message}") from exc
        except anthropic.RateLimitError as exc:
            raise SynthesisError("Claude rate limit reached after retries") from exc
        except anthropic.APIStatusError as exc:
            raise SynthesisError(f"Claude API error {exc.status_code}") from exc
        except anthropic.APIConnectionError as exc:
            raise SynthesisError("Could not reach the Claude API") from exc

        if response.stop_reason == "refusal":
            raise SynthesisError("Claude declined to answer this request")
        if response.stop_reason == "max_tokens":
            raise SynthesisError("Claude response was truncated (max_tokens)")
        text = next((b.text for b in response.content if b.type == "text"), "")
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            raise SynthesisError("Claude returned malformed JSON") from exc

    def synthesize(self, query: str, papers: list[dict]) -> dict[str, dict]:
        result = self._call(SYNTHESIS_SYSTEM, synthesis_user_message(query, papers), SYNTHESIS_SCHEMA,
                            max_tokens=16000, effort=self.effort)
        return {p["paper_key"]: p for p in result.get("papers", [])}

    def paraphrase_query(self, query: str, n: int = 3) -> list[str]:
        result = self._call(PARAPHRASE_SYSTEM, paraphrase_user_message(query, n), PARAPHRASE_SCHEMA,
                            max_tokens=2000, effort="low")
        return [r.strip() for r in result.get("rewrites", []) if r.strip()][:n]
