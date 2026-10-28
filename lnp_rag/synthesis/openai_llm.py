"""OpenAI provider for evidence synthesis and (optional) query paraphrasing.

Uses Chat Completions with strict JSON-schema structured outputs, so the
response always matches the schema that ``synthesis.grounding`` validates.
"""
from __future__ import annotations

import json
import logging

import openai

from lnp_rag.config import Settings
from lnp_rag.synthesis.prompts import (
    CRITIC_SCHEMA,
    CRITIC_SYSTEM,
    PARAPHRASE_SCHEMA,
    PARAPHRASE_SYSTEM,
    SYNTHESIS_SCHEMA,
    SYNTHESIS_SYSTEM,
    SynthesisError,
    critic_user_message,
    paraphrase_user_message,
    synthesis_user_message,
)

log = logging.getLogger(__name__)


class OpenAIClient:
    name = "openai"

    def __init__(self, settings: Settings):
        if not settings.openai_api_key:
            raise SynthesisError("OPENAI_API_KEY is not set")
        self._client = openai.OpenAI(
            api_key=settings.openai_api_key.get_secret_value(),
            max_retries=settings.llm_max_retries,
            timeout=settings.llm_timeout_s,
        )
        self.model = settings.openai_model
        self.reasoning_effort = settings.openai_reasoning_effort.strip()

    def _call(self, system: str, user: str, schema: dict, schema_name: str, max_tokens: int,
              reasoning_effort: str = "") -> dict:
        kwargs: dict = {}
        effort = reasoning_effort or self.reasoning_effort
        if effort:  # only reasoning models accept this; leave empty for others
            kwargs["reasoning_effort"] = effort
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                response_format={
                    "type": "json_schema",
                    "json_schema": {"name": schema_name, "schema": schema, "strict": True},
                },
                max_completion_tokens=max_tokens,
                **kwargs,
            )
        except openai.AuthenticationError as exc:
            raise SynthesisError("OpenAI authentication failed - check OPENAI_API_KEY") from exc
        except openai.NotFoundError as exc:
            raise SynthesisError(f"OpenAI model '{self.model}' not available to this key") from exc
        except openai.BadRequestError as exc:
            raise SynthesisError(f"OpenAI rejected the request: {exc.message}") from exc
        except openai.RateLimitError as exc:
            raise SynthesisError("OpenAI rate limit or quota reached after retries") from exc
        except openai.APIStatusError as exc:
            raise SynthesisError(f"OpenAI API error {exc.status_code}") from exc
        except openai.APIConnectionError as exc:
            raise SynthesisError("Could not reach the OpenAI API") from exc

        choice = response.choices[0]
        if getattr(choice.message, "refusal", None):
            raise SynthesisError("The OpenAI model declined to answer this request")
        if choice.finish_reason == "length":
            raise SynthesisError("OpenAI response was truncated (max_completion_tokens)")
        try:
            return json.loads(choice.message.content or "")
        except json.JSONDecodeError as exc:
            raise SynthesisError("OpenAI returned malformed JSON") from exc

    def synthesize(self, query: str, papers: list[dict]) -> dict[str, dict]:
        result = self._call(SYNTHESIS_SYSTEM, synthesis_user_message(query, papers), SYNTHESIS_SCHEMA,
                            "paper_assessments", max_tokens=16000)
        return {p["paper_key"]: p for p in result.get("papers", [])}

    def review(self, query: str, folder: str, papers: list[dict]) -> dict[str, dict]:
        result = self._call(CRITIC_SYSTEM, critic_user_message(query, folder, papers), CRITIC_SCHEMA,
                            "critic_verdicts", max_tokens=16000)
        return {p["paper_key"]: p for p in result.get("papers", [])}

    def paraphrase_query(self, query: str, n: int = 3) -> list[str]:
        result = self._call(PARAPHRASE_SYSTEM, paraphrase_user_message(query, n), PARAPHRASE_SCHEMA,
                            "query_rewrites", max_tokens=2000, reasoning_effort="low" if self.reasoning_effort else "")
        return [r.strip() for r in result.get("rewrites", []) if r.strip()][:n]
