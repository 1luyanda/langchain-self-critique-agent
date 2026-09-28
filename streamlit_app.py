"""Streamlit UI for the Day 7 bounded self-critique agent.

All agent logic lives in self_critique_agent.py. This file only collects a
question, calls that module once, and renders the result.
"""

from __future__ import annotations

import streamlit as st

from self_critique_agent import (
    BaselineRun,
    MissingAzureConfig,
    SelfCritiqueAgent,
    SelfCritiqueRun,
    TokenUsage,
    azure_config_status,
    format_token_display,
    load_project_env,
)

EXAMPLE_QUESTIONS = [
    "Who directed Inception?",
    "When did Neil Armstrong first walk on Mars?",
    "Who directed the first Rush Hour film, who was the lead actor, and in which year was it released?",
]

STATUS_LABEL = {
    "verified": "verified",
    "contradicted": "contradicted",
    "insufficient_evidence": "insufficient evidence",
    "not_applicable": "not applicable",
}


def public_runtime_error(exc: BaseException) -> str:
    """User-facing error text. Does not include keys, endpoints, or raw payloads."""
    kind = type(exc).__name__
    if isinstance(exc, MissingAzureConfig):
        return str(exc)
    if kind in {"AuthenticationError", "PermissionDeniedError"}:
        return (
            "Azure OpenAI authentication failed. Check AZURE_OPENAI_API_KEY in "
            ".env. The key is not displayed here."
        )
    if kind in {"NotFoundError", "NotFound"}:
        return (
            "Azure OpenAI deployment was not found. "
            "AZURE_OPENAI_DEPLOYMENT must be a chat deployment (for example gpt-4.1)."
        )
    if kind in {"RateLimitError"}:
        return "Azure OpenAI rate limit was hit. Wait a moment and try again."
    if kind in {"APITimeoutError", "APIConnectionError", "ConnectError"}:
        return (
            "Could not reach Azure OpenAI. Check the network connection. "
            "The endpoint value is not displayed here."
        )
    if "wikipedia" in str(exc).lower() and kind == "ImportError":
        return (
            "The wikipedia package is missing in this environment. "
            "Run `pip install wikipedia` in this project's virtual environment."
        )
    return (
        f"The request failed ({kind}). Check Azure configuration and Wikipedia access. "
        "Secret values are never shown."
    )


def render_tokens(usage: TokenUsage) -> None:
    st.write("**Tokens:**", format_token_display(usage))


def render_performance(latency_ms: float, usage: TokenUsage, wikipedia_lookups: int | None = None) -> None:
    st.metric("Latency", f"{latency_ms:.0f} ms")
    render_tokens(usage)
    if wikipedia_lookups is not None:
        st.write("**Wikipedia lookups:**", wikipedia_lookups)


def render_claims(run: SelfCritiqueRun) -> None:
    if not run.claims:
        st.info("No claims were extracted from the initial answer.")
        return
    for claim in run.claims:
        verifiable = "verifiable" if claim.is_verifiable else "not independently verifiable"
        st.markdown(f"**{claim.claim_id}** — {verifiable}")
        st.write(claim.claim_text)
        st.caption(claim.reason)


def render_verifications(run: SelfCritiqueRun) -> None:
    if not run.verification_results:
        st.info("No verification results. Wikipedia was not queried.")
        return
    for item in run.verification_results:
        label = STATUS_LABEL.get(item.status, item.status)
        st.markdown(f"**{item.claim_id}** — `{label}`")
        st.write(item.claim_text)
        if item.source_query:
            st.caption(f"Wikipedia query: {item.source_query}")
        st.write(item.evidence)
        st.divider()


@st.cache_resource
def load_agent() -> SelfCritiqueAgent:
    return SelfCritiqueAgent.from_env()


def get_agent() -> SelfCritiqueAgent:
    return load_agent()


st.set_page_config(page_title="Day 7 - Self-Critiquing AI Agent", layout="wide")
load_project_env()

st.title("Day 7 - Self-Critiquing AI Agent")

st.markdown(
    """
This app runs **one** factual QA pipeline with a **bounded** self-check.
It is not several agents talking to each other.

1. **Answer** — write an initial reply. No Wikipedia.
2. **Decompose** — extract only independently checkable factual claims.
3. **Verify** — look up each verifiable claim on Wikipedia once.
4. **Revise** — correct or remove contradicted claims, flag insufficient evidence, keep verified claims. **Stop.** There is no second loop.
"""
)

with st.sidebar:
    st.header("Example questions")
    st.caption("Click one to fill the question box.")
    for index, example in enumerate(EXAMPLE_QUESTIONS):
        if st.button(example, key=f"example_{index}"):
            st.session_state["question_text"] = example
    st.divider()
    st.subheader("Azure configuration")
    st.caption("Names only. Values are never printed.")
    for name, status in azure_config_status().items():
        st.write(f"{name}: **{status}**")

if "question_text" not in st.session_state:
    st.session_state["question_text"] = ""

question = st.text_area(
    "Factual question",
    key="question_text",
    height=120,
    placeholder="Ask a factual question…",
)

col_verify, col_baseline = st.columns(2)
verify_clicked = col_verify.button("Verify my answer", type="primary")
baseline_clicked = col_baseline.button("Run baseline only")

if verify_clicked or baseline_clicked:
    cleaned = (question or "").strip()
    if not cleaned:
        st.error("Enter a factual question, or choose an example in the sidebar.")
    else:
        try:
            agent = get_agent()
        except MissingAzureConfig as exc:
            st.error(public_runtime_error(exc))
        except ImportError as exc:
            st.error(public_runtime_error(exc))
        except Exception as exc:
            st.error(public_runtime_error(exc))
        else:
            spinner_label = (
                "Running Answer → Decompose → Verify → Revise (one pass)…"
                if verify_clicked
                else "Running baseline answer (no Wikipedia)…"
            )
            try:
                with st.spinner(spinner_label):
                    if verify_clicked:
                        result: SelfCritiqueRun | BaselineRun = agent.run_self_critique(cleaned)
                    else:
                        result = agent.baseline_answer(cleaned)
            except Exception as exc:
                st.error(public_runtime_error(exc))
            else:
                if isinstance(result, BaselineRun):
                    st.success("Baseline finished. Verification was not run.")
                    with st.expander("1. Initial answer", expanded=True):
                        st.write(result.original_answer)
                    with st.expander("2. Claims extracted"):
                        st.info("Skipped. Use **Verify my answer** to extract claims.")
                    with st.expander("3. Wikipedia verification"):
                        st.info("Skipped. Baseline does not call Wikipedia.")
                    with st.expander("4. Revised answer"):
                        st.info("Skipped. There is no revision without verification.")
                    with st.expander("Performance", expanded=True):
                        render_performance(result.latency_ms, result.token_usage)
                else:
                    st.success("Bounded self-critique finished (one revision only).")
                    with st.expander("1. Initial answer", expanded=True):
                        st.write(result.original_answer)
                    with st.expander("2. Claims extracted", expanded=True):
                        render_claims(result)
                    with st.expander("3. Wikipedia verification", expanded=True):
                        render_verifications(result)
                    with st.expander("4. Revised answer", expanded=True):
                        st.write(result.revised_answer)
                    with st.expander("Performance", expanded=True):
                        render_performance(
                            result.latency_ms,
                            result.token_usage,
                            wikipedia_lookups=result.wikipedia_lookups,
                        )
