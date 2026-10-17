"""LLM screening of retrieved candidates (OpenAI or Anthropic)."""

from __future__ import annotations

import hashlib
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import anthropic
import openai
from tqdm import tqdm

DEFAULT_MODELS = {"openai": "gpt-5-mini", "anthropic": "claude-opus-5-5"}

VERDICT_RANK = {"include": 2, "borderline": 1, "exclude": 0}

RESULT_SCHEMA = {
    "type": "object",
    "properties": {
        "score": {"type": "number"},
        "verdict": {"type": "string", "enum": ["include", "borderline", "exclude"]},
        "study_type": {"type": "string", "enum": ["in_vitro", "in_vivo", "both", "unclassified"]},
        "data_types": {"type": "array", "items": {"type": "string", "enum": [
            "time_course", "dose_response", "uptake_quantification", "intracellular_accumulation",
            "endosomal_escape", "uptake_pathway",
        ]}},
        "rationale": {"type": "string"},
    },
    "required": ["score", "verdict", "study_type", "data_types", "rationale"],
    "additionalProperties": False,
}


def _system_prompt(
    step_name: str, step_description: str, strict_kinetic: bool, guidance: str = ""
) -> str:
    kinetic_clause = (
        "The paper MUST contain time-resolved/kinetic data (rates, half-lives, "
        "time courses, t1/2). Papers without explicit time-dependent measurements "
        "are 'exclude'. EXCEPTION: if the user message marks the paper as a "
        "REVIEW ARTICLE, this kinetic requirement does not apply — judge it only "
        "on topical relevance to the step, since a review synthesizes other "
        "papers' data rather than reporting its own."
        if strict_kinetic
        else "Time-resolved data is a strong plus but not strictly required."
    )
    return (
        "You are a biomedical literature screener for a Lipid Nanoparticle (LNP) "
        "delivery-journey review. Decide whether the paper is on-topic for the "
        f"step '{step_name}' ({step_description}).\n\n"
        "A paper is 'include' only if it (a) is about lipid nanoparticles "
        "specifically (not generic nanoparticles, liposomes for other purposes, "
        f"polymeric NPs, etc.), and (b) studies the '{step_name}' "
        f"step of the LNP journey. {kinetic_clause}\n\n"
        + (f"Step-specific guidance: {guidance}\n\n" if guidance else "")
        + "Evidence may come from the abstract OR, when provided, the figure/table "
        "captions, Results-text excerpts and supplementary material taken from "
        "the paper's full text. Abstracts often omit experimental detail that "
        "only appears in figures or the Results, so treat measurements described "
        "there (e.g. 'uptake measured at 1, 6 and 24 h by flow cytometry', "
        "'at 48 h post-transfection, 91.8% of cells were GFP-positive') as valid "
        "evidence, including valid time-resolved/kinetic evidence. A measurement "
        "at two or more time points counts as a time course.\n\n"
        "Separately, classify the paper's study type (independent of the verdict):\n"
        "- 'in_vitro': experiments in cultured cells, cell lines, primary cells, "
        "organoids, ex vivo tissue, or biological fluids such as plasma/serum "
        "outside a living organism.\n"
        "- 'in_vivo': experiments in living animals (mice, rats, NHPs, zebrafish, "
        "etc.) or humans/patients, including dosing, biodistribution, efficacy or "
        "expression measured in the organism. Infer this from context even if no "
        "species is named (e.g. delivery to the retina, liver, or tumors after "
        "injection).\n"
        "- 'both': the paper reports both in vitro and in vivo experiments.\n"
        "- 'unclassified': no biological experiments (purely computational, "
        "simulation, physicochemical or formulation characterization), or too "
        "little information to tell (e.g. title only).\n"
        "For a REVIEW ARTICLE, classify by the kind of evidence it mainly "
        "discusses. With no abstract, use the title and captions if they make "
        "the study type clear; otherwise 'unclassified'.\n\n"
        "Also list the quantitative data types the paper itself reports (empty list if none):\n"
        "- 'time_course': a cellular measurement at two or more time points.\n"
        "- 'dose_response': a cellular measurement at two or more LNP/cargo doses or concentrations.\n"
        "- 'uptake_quantification': cell-associated LNP or cargo quantified (MFI, % positive cells, "
        "ng or copies per cell...).\n"
        "- 'intracellular_accumulation': intracellular amount or localization of LNP/cargo quantified "
        "(e.g. colocalization with endosomes/lysosomes, cytosolic cargo).\n"
        "- 'endosomal_escape': endosomal escape or endosomal disruption measured (galectin recruitment, "
        "escape efficiency, cytosolic release).\n"
        "- 'uptake_pathway': endocytic pathway dissected (inhibitors, knockdown, temperature).\n\n"
        "Return JSON ONLY in this exact shape:\n"
        '{"score": <float 0..1>, "verdict": "include"|"borderline"|"exclude", '
        '"study_type": "in_vitro"|"in_vivo"|"both"|"unclassified", "data_types": [...], '
        '"rationale": "<<=240 chars, plain English, cite specific evidence from '
        'the abstract or captions>"}'
    )


def build_user_message(paper: dict, excerpts: list[str]) -> str:
    parts = []
    if paper["is_review"]:
        parts.append("NOTE: This paper is a REVIEW ARTICLE.")
    parts.append(f"Title: {paper['title']}")
    parts.append(f"Abstract: {paper['abstract'] or '(no abstract found)'}")
    if paper["captions"]:
        caps = "\n".join(f"- {c}" for c in paper["captions"])[:5000]
        parts.append(f"Figure/table captions:\n{caps}")
    if paper.get("supplement"):
        parts.append(f"Supplementary material (excerpt):\n{paper['supplement'][:2500]}")
    seen = set()
    texts = []
    for e in excerpts + ([paper["results"][:3000]] if paper["results"] else []):
        if e[:200] not in seen:
            seen.add(e[:200])
            texts.append(e)
    if texts:
        parts.append("Results-text excerpts (full text):\n" + "\n---\n".join(texts)[:7000])
    return "\n\n".join(parts)


class Screener:
    """Calls the LLM once per paper; verdicts are cached on disk keyed by prompt + paper content.

    `schema` is the JSON schema the model must answer in (the screening verdict by default).
    """

    def __init__(
        self, system_prompt: str, cache_path: Path, provider: str = "openai",
        model: str | None = None, effort: str = "low", schema: dict = RESULT_SCHEMA,
    ):
        self.provider = provider
        self.model = model or DEFAULT_MODELS[provider]
        if provider == "anthropic":
            self.client = anthropic.Anthropic(max_retries=6)
        else:
            self.client = openai.OpenAI(max_retries=6)
        self.system = system_prompt
        self.effort = effort
        self.schema = schema
        self.cache_path = cache_path
        self._lock = threading.Lock()
        self.cache: dict[str, dict] = {}
        if cache_path.exists():
            with open(cache_path) as f:
                for line in f:
                    row = json.loads(line)
                    self.cache[row["key"]] = row["result"]

    def _key(self, user_msg: str) -> str:
        schema = "" if self.schema is RESULT_SCHEMA else json.dumps(self.schema, sort_keys=True)
        return hashlib.sha256(f"{self.model}\0{self.system}\0{schema}{user_msg}".encode()).hexdigest()

    def _call_anthropic(self, user_msg: str) -> str | dict:
        try:
            response = self.client.beta.messages.create(
                model=self.model,
                max_tokens=8000,
                system=self.system,
                messages=[{"role": "user", "content": user_msg}],
                thinking={"type": "adaptive"},
                output_config={"effort": self.effort, "format": {"type": "json_schema", "schema": self.schema}},
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",
            )
        except (anthropic.AuthenticationError, anthropic.PermissionDeniedError):
            raise  # bad credentials affect every paper: stop the run
        except anthropic.BadRequestError as e:
            return {"error": f"bad request: {e.message}"}
        except anthropic.APIStatusError as e:
            return {"error": f"API error {e.status_code}: {e.message}"}
        except anthropic.APIConnectionError as e:
            return {"error": f"connection error: {e}"}

        if response.stop_reason == "refusal":
            return {"error": "model declined to screen this paper"}
        if response.stop_reason == "max_tokens":
            return {"error": "response truncated (max_tokens)"}
        return next((b.text for b in response.content if b.type == "text"), "")

    def _call_openai(self, user_msg: str) -> str | dict:
        kwargs = {}
        if self.model.startswith(("gpt-5", "o")):  # reasoning models
            kwargs["reasoning_effort"] = self.effort if self.effort in ("low", "medium", "high") else "high"
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "system", "content": self.system}, {"role": "user", "content": user_msg}],
                max_completion_tokens=8000,
                response_format={
                    "type": "json_schema",
                    "json_schema": {"name": "result", "strict": True, "schema": self.schema},
                },
                **kwargs,
            )
        except (openai.AuthenticationError, openai.PermissionDeniedError, openai.NotFoundError):
            raise  # bad key or unknown model affects every paper: stop the run
        except openai.BadRequestError as e:
            return {"error": f"bad request: {e.message}"}
        except openai.APIStatusError as e:
            return {"error": f"API error {e.status_code}: {e.message}"}
        except openai.APIConnectionError as e:
            return {"error": f"connection error: {e}"}

        choice = response.choices[0]
        if choice.message.refusal:
            return {"error": f"model declined: {choice.message.refusal[:200]}"}
        if choice.finish_reason == "length":
            return {"error": "response truncated (max tokens)"}
        return choice.message.content or ""

    def screen_one(self, user_msg: str) -> dict:
        key = self._key(user_msg)
        if key in self.cache:
            return self.cache[key]
        call = self._call_anthropic if self.provider == "anthropic" else self._call_openai
        text = call(user_msg)
        if isinstance(text, dict):
            return text
        try:
            result = json.loads(text)
        except json.JSONDecodeError:
            return {"error": f"unparseable response: {text[:200]}"}
        if "score" in result:
            result["score"] = min(1.0, max(0.0, float(result["score"])))

        with self._lock:
            self.cache[key] = result
            with open(self.cache_path, "a") as f:
                f.write(json.dumps({"key": key, "result": result}) + "\n")
        return result

    def screen_many(self, user_msgs: list[str], workers: int = 8, on_progress=None) -> list[dict]:
        results = []
        with ThreadPoolExecutor(workers) as pool:
            for res in tqdm(pool.map(self.screen_one, user_msgs), total=len(user_msgs), desc="screening"):
                results.append(res)
                if on_progress:
                    on_progress(len(results), len(user_msgs))
        return results
