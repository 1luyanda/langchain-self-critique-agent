"""Reusable Day 7 bounded self-critique agent (Answer -> Decompose -> Verify -> Revise).

This module is the source of truth for the Streamlit app. It mirrors the bonus
notebook pipeline: one pass only, Wikipedia once per verifiable claim, one revision.
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from dotenv import load_dotenv
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, ConfigDict, Field

DAY7_DIR = Path(__file__).resolve().parent

REQUIRED_AZURE = (
    "AZURE_OPENAI_ENDPOINT",
    "AZURE_OPENAI_API_KEY",
    "AZURE_OPENAI_API_VERSION",
    "AZURE_OPENAI_DEPLOYMENT",
)

WIKI_MAX_CHARS = 700

ANSWER_SYSTEM = (
    "You are a factual question-answering assistant. "
    "Answer in 2-6 concise sentences. "
    "Do not use tools. Do not mention Wikipedia or verification. "
    "Do not include hidden reasoning."
)

DECOMPOSE_SYSTEM = (
    "Extract independently checkable factual claims from the assistant answer. "
    "Do not treat opinions, advice, subjective wording, hedged language, or generic "
    "statements as verifiable facts. Set is_verifiable true only for specific "
    "checkable facts (names, dates, numbers, places, titles). "
    "Return Claim objects only. Do not include hidden reasoning."
)

VERIFY_SYSTEM = (
    "You compare one factual claim to compact Wikipedia evidence. "
    "status must be exactly one of: verified, contradicted, insufficient_evidence. "
    "Mark verified ONLY if the evidence clearly supports the claim. "
    "Mark contradicted if the evidence clearly conflicts. "
    "Mark insufficient_evidence if the lookup is empty, failed, unrelated, or too vague. "
    "Copy a short evidence snippet into the evidence field. "
    "Do not invent sources. Do not include hidden reasoning."
)

REVISE_SYSTEM = (
    "Revise the original answer using the verification results. "
    "Preserve claims with status verified. "
    "Correct or remove claims with status contradicted; do not keep the false wording. "
    "Clearly flag claims with status insufficient_evidence (say evidence was insufficient). "
    "Do not add new unsupported facts. Do not call tools. "
    "This is the only revision; do not propose another loop. "
    "Return the revised answer only."
)


class MissingAzureConfig(RuntimeError):
    """Raised when required Azure OpenAI environment names are missing."""


class Claim(BaseModel):
    """One statement taken from the initial answer."""

    model_config = ConfigDict(extra="forbid")

    claim_id: str = Field(description="Stable id such as C1, C2.")
    claim_text: str = Field(description="The claim in the answer's own wording, shortened if needed.")
    is_verifiable: bool = Field(
        description="True only if this is an independently checkable factual claim."
    )
    reason: str = Field(
        description="Why it is or is not independently verifiable (not a chain of thought)."
    )


class ClaimList(BaseModel):
    model_config = ConfigDict(extra="forbid")

    claims: list[Claim]


class VerificationResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    claim_id: str
    claim_text: str
    status: Literal["verified", "contradicted", "insufficient_evidence", "not_applicable"]
    evidence: str = Field(description="Compact Wikipedia snippet or a safe empty-result note.")
    source_query: str = Field(description="Query sent to Wikipedia.")


class TokenUsage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    metadata_available: bool = False


class SelfCritiqueRun(BaseModel):
    """Observable record of one bounded Answer -> Decompose -> Verify -> Revise pass."""

    model_config = ConfigDict(extra="forbid")

    question: str
    original_answer: str
    claims: list[Claim]
    verification_results: list[VerificationResult]
    revised_answer: str
    latency_ms: float
    token_usage: TokenUsage
    step_log: str
    wikipedia_lookups: int = 0


class BaselineRun(BaseModel):
    """One answer with no Wikipedia and no revision."""

    model_config = ConfigDict(extra="forbid")

    question: str
    original_answer: str
    latency_ms: float
    token_usage: TokenUsage


def _clean_env(value: str | None) -> str:
    """Strip whitespace and accidental quotes from an environment value."""
    return (value or "").strip().strip('"').strip("'")


def env_status(key: str) -> str:
    """Return SET or MISSING so callers can log config without printing secrets."""
    return "SET" if _clean_env(os.getenv(key)) else "MISSING"


def azure_config_status() -> dict[str, str]:
    """Map each required Azure name to SET or MISSING."""
    return {key: env_status(key) for key in REQUIRED_AZURE}


def missing_azure_names() -> list[str]:
    return [key for key in REQUIRED_AZURE if env_status(key) == "MISSING"]


def azure_v1_base_url(endpoint: str) -> str:
    """Build the Azure OpenAI v1 base URL that ChatOpenAI expects."""
    cleaned = _clean_env(endpoint).rstrip("/")
    if cleaned.endswith("/openai/v1"):
        return cleaned + "/"
    return cleaned + "/openai/v1/"


def load_project_env() -> Path | None:
    """Load this project's .env. Never prints secret values."""
    env_path = DAY7_DIR / ".env"
    if env_path.is_file():
        load_dotenv(env_path, override=True)
        return env_path
    load_dotenv()
    return None


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def message_text(msg: Any) -> str:
    content = getattr(msg, "content", msg)
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                parts.append(str(block.get("text") or ""))
            else:
                text = getattr(block, "text", None)
                if text:
                    parts.append(str(text))
        return "\n".join(p for p in parts if p).strip()
    return str(content).strip()


def extract_token_usage(message: Any) -> TokenUsage:
    meta = getattr(message, "usage_metadata", None)
    if isinstance(meta, dict):
        inp = meta.get("input_tokens")
        out = meta.get("output_tokens")
        tot = meta.get("total_tokens")
        if inp is not None or out is not None or tot is not None:
            inp_i = int(inp) if inp is not None else None
            out_i = int(out) if out is not None else None
            tot_i = int(tot) if tot is not None else None
            if tot_i is None and inp_i is not None and out_i is not None:
                tot_i = inp_i + out_i
            return TokenUsage(
                input_tokens=inp_i,
                output_tokens=out_i,
                total_tokens=tot_i,
                metadata_available=True,
            )
    rm = getattr(message, "response_metadata", None) or {}
    tu = rm.get("token_usage") or rm.get("usage") or {}
    if isinstance(tu, dict) and tu:
        inp = tu.get("prompt_tokens") or tu.get("input_tokens")
        out = tu.get("completion_tokens") or tu.get("output_tokens")
        tot = tu.get("total_tokens")
        if inp is not None or out is not None or tot is not None:
            inp_i = int(inp) if inp is not None else None
            out_i = int(out) if out is not None else None
            tot_i = int(tot) if tot is not None else None
            if tot_i is None and inp_i is not None and out_i is not None:
                tot_i = inp_i + out_i
            return TokenUsage(
                input_tokens=inp_i,
                output_tokens=out_i,
                total_tokens=tot_i,
                metadata_available=True,
            )
    return TokenUsage(metadata_available=False)


def add_usage(left: TokenUsage, right: TokenUsage) -> TokenUsage:
    if not left.metadata_available and not right.metadata_available:
        return TokenUsage(metadata_available=False)
    if not left.metadata_available:
        return right
    if not right.metadata_available:
        return left

    def add_opt(a: int | None, b: int | None) -> int | None:
        if a is None and b is None:
            return None
        return (a or 0) + (b or 0)

    inp = add_opt(left.input_tokens, right.input_tokens)
    out = add_opt(left.output_tokens, right.output_tokens)
    tot = add_opt(left.total_tokens, right.total_tokens)
    if tot is None and inp is not None and out is not None:
        tot = inp + out
    return TokenUsage(
        input_tokens=inp,
        output_tokens=out,
        total_tokens=tot,
        metadata_available=True,
    )


def invoke_plain(model: ChatOpenAI, system: str, user: str) -> tuple[str, TokenUsage]:
    msg = model.invoke([SystemMessage(content=system), HumanMessage(content=user)])
    return message_text(msg), extract_token_usage(msg)


def invoke_structured(
    model: ChatOpenAI,
    schema: type[BaseModel],
    system: str,
    user: str,
) -> tuple[Any, TokenUsage]:
    runnable = model.with_structured_output(schema, include_raw=True)
    out = runnable.invoke([SystemMessage(content=system), HumanMessage(content=user)])
    if isinstance(out, dict):
        parsed = out.get("parsed")
        raw = out.get("raw")
        usage = extract_token_usage(raw) if raw is not None else TokenUsage(metadata_available=False)
        if parsed is None:
            raise ValueError(
                out.get("parsing_error") or f"structured output for {schema.__name__} was empty"
            )
        return parsed, usage
    if isinstance(out, schema):
        return out, TokenUsage(metadata_available=False)
    raise ValueError(f"Unexpected structured output type: {type(out)!r}")


def format_token_display(usage: TokenUsage) -> str:
    """Human-readable token line. Never invent counts."""
    if not usage.metadata_available:
        return "Not available"
    parts = [
        f"input={usage.input_tokens}",
        f"output={usage.output_tokens}",
        f"total={usage.total_tokens}",
    ]
    return ", ".join(parts)


class SelfCritiqueAgent:
    """One application: Answer, then one bounded verify-and-revise pass."""

    def __init__(self, model: ChatOpenAI, wikipedia_api: WikipediaAPIWrapper) -> None:
        self.model = model
        self.wikipedia_api = wikipedia_api

    @classmethod
    def from_env(cls) -> SelfCritiqueAgent:
        """Build the agent from this folder's .env. Does not call Azure until a question is run."""
        load_project_env()
        missing = missing_azure_names()
        if missing:
            raise MissingAzureConfig(
                "Azure OpenAI is not configured. Missing: "
                + ", ".join(missing)
                + ". Copy .env.example to .env and fill the four names. "
                "Secret values are never displayed."
            )
        deployment_name = _clean_env(os.getenv("AZURE_OPENAI_DEPLOYMENT"))
        if "embed" in deployment_name.lower():
            raise MissingAzureConfig(
                "AZURE_OPENAI_DEPLOYMENT must be a chat deployment, not embeddings."
            )
        api_key = _clean_env(os.getenv("AZURE_OPENAI_API_KEY"))
        endpoint = _clean_env(os.getenv("AZURE_OPENAI_ENDPOINT"))
        model = ChatOpenAI(
            model=deployment_name,
            api_key=api_key,
            base_url=azure_v1_base_url(endpoint),
            default_headers={"api-key": api_key},
            temperature=0,
        )
        try:
            wikipedia_api = WikipediaAPIWrapper(
                top_k_results=1,
                doc_content_chars_max=WIKI_MAX_CHARS,
                lang="en",
            )
        except ImportError as exc:
            raise ImportError(
                "Could not import the wikipedia package. "
                "Install it with `pip install wikipedia` into this project's .venv."
            ) from exc
        return cls(model=model, wikipedia_api=wikipedia_api)

    def lookup_wikipedia_evidence(self, query: str) -> str:
        """Look up compact Wikipedia evidence. Empty or failed lookups are safe strings."""
        cleaned = (query or "").strip()
        if not cleaned:
            return "NO_WIKIPEDIA_RESULT: empty query. Do not treat this as supporting evidence."
        try:
            raw = self.wikipedia_api.run(cleaned[:200])
        except Exception as exc:
            return (
                f"NO_WIKIPEDIA_RESULT: lookup failed ({type(exc).__name__}). "
                "Do not treat this as supporting evidence."
            )
        text = (raw or "").strip()
        if not text or "No good Wikipedia Search Result was found" in text:
            return (
                "NO_WIKIPEDIA_RESULT: no usable page summary. "
                "Do not treat this as supporting evidence."
            )
        return text[:WIKI_MAX_CHARS]

    def baseline_answer(self, question: str) -> BaselineRun:
        """Initial answer with no verification loop."""
        t0 = time.perf_counter()
        text, usage = invoke_plain(self.model, ANSWER_SYSTEM, question)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return BaselineRun(
            question=question,
            original_answer=text,
            latency_ms=elapsed_ms,
            token_usage=usage,
        )

    def decompose_claims(self, question: str, answer: str) -> tuple[list[Claim], TokenUsage]:
        user = f"Question:\n{question}\n\nAnswer:\n{answer}"
        parsed, usage = invoke_structured(self.model, ClaimList, DECOMPOSE_SYSTEM, user)
        claims = list(parsed.claims)
        for i, claim in enumerate(claims, start=1):
            if not claim.claim_id:
                claim.claim_id = f"C{i}"
        return claims, usage

    def verify_one_claim(self, claim: Claim) -> tuple[VerificationResult, TokenUsage, bool]:
        """Verify one claim. Wikipedia is called at most once. No retry loop."""
        if not claim.is_verifiable:
            result = VerificationResult(
                claim_id=claim.claim_id,
                claim_text=claim.claim_text,
                status="not_applicable",
                evidence="Not an independently checkable factual claim.",
                source_query="",
            )
            return result, TokenUsage(metadata_available=False), False

        source_query = claim.claim_text[:180]
        wiki_text = self.lookup_wikipedia_evidence(source_query)
        if wiki_text.startswith("NO_WIKIPEDIA_RESULT"):
            result = VerificationResult(
                claim_id=claim.claim_id,
                claim_text=claim.claim_text,
                status="insufficient_evidence",
                evidence=wiki_text,
                source_query=source_query,
            )
            return result, TokenUsage(metadata_available=False), True

        user = (
            f"Claim id: {claim.claim_id}\n"
            f"Claim: {claim.claim_text}\n\n"
            f"Wikipedia evidence:\n{wiki_text}\n\n"
            f"source_query: {source_query}"
        )
        parsed, usage = invoke_structured(self.model, VerificationResult, VERIFY_SYSTEM, user)
        status = parsed.status
        if not (parsed.evidence or "").strip() and status == "verified":
            status = "insufficient_evidence"
        result = VerificationResult(
            claim_id=claim.claim_id,
            claim_text=claim.claim_text,
            status=status if status != "not_applicable" else "insufficient_evidence",
            evidence=(parsed.evidence or wiki_text)[:WIKI_MAX_CHARS],
            source_query=source_query,
        )
        return result, usage, True

    def revise_answer(
        self,
        question: str,
        original: str,
        verifications: list[VerificationResult],
    ) -> tuple[str, TokenUsage]:
        payload = json.dumps([v.model_dump() for v in verifications], indent=2)
        user = (
            f"Question:\n{question}\n\n"
            f"Original answer:\n{original}\n\n"
            f"Verification results (JSON):\n{payload}"
        )
        return invoke_plain(self.model, REVISE_SYSTEM, user)

    def run_self_critique(self, question: str) -> SelfCritiqueRun:
        """One bounded Answer -> Decompose -> Verify -> Revise pass. No second revision."""
        log_lines: list[str] = []
        usage = TokenUsage(metadata_available=False)
        wiki_lookups = 0
        t0 = time.perf_counter()

        baseline = self.baseline_answer(question)
        original = baseline.original_answer
        usage = add_usage(usage, baseline.token_usage)
        log_lines.append(f"[{utc_now()}] original answer -> {original}")

        claims, decomp_usage = self.decompose_claims(question, original)
        usage = add_usage(usage, decomp_usage)
        claim_dump = json.dumps([c.model_dump() for c in claims], indent=2)
        log_lines.append(f"[{utc_now()}] extracted claims -> {claim_dump}")

        verifications: list[VerificationResult] = []
        for claim in claims:
            result, v_usage, used_wiki = self.verify_one_claim(claim)
            usage = add_usage(usage, v_usage)
            if used_wiki:
                wiki_lookups += 1
            verifications.append(result)
        ver_dump = json.dumps([v.model_dump() for v in verifications], indent=2)
        log_lines.append(f"[{utc_now()}] verification results -> {ver_dump}")

        # Bounded: a single revision. Do not call run_self_critique or revise_answer again.
        revised, rev_usage = self.revise_answer(question, original, verifications)
        usage = add_usage(usage, rev_usage)
        log_lines.append(f"[{utc_now()}] revised answer -> {revised}")

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return SelfCritiqueRun(
            question=question,
            original_answer=original,
            claims=claims,
            verification_results=verifications,
            revised_answer=revised,
            latency_ms=elapsed_ms,
            token_usage=usage,
            step_log="\n".join(log_lines),
            wikipedia_lookups=wiki_lookups,
        )
