# Bounded Self-Critique Agent

A factual Q&A agent that answers once, splits the answer into checkable claims, verifies those claims against Wikipedia, then revises the answer a single time. There is no second revision loop.

## What is implemented

- Pipeline: Answer → Decompose → Verify → Revise (once)
- Wikipedia lookup only during Verify
- Streamlit UI that collects a question and renders the stages
- Azure OpenAI configuration loaded from `.env` (names shown as SET/MISSING, values never printed)

## Technology stack

Python, LangChain, Azure OpenAI, Wikipedia API wrapper, Streamlit, Pydantic.

## Installation (Windows PowerShell)

```powershell
cd path\to\langchain-self-critique-agent
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
```

Fill `.env`. Do not commit it.

| Variable | Required |
| --- | --- |
| `AZURE_OPENAI_ENDPOINT` | Yes |
| `AZURE_OPENAI_API_KEY` | Yes |
| `AZURE_OPENAI_API_VERSION` | Yes |
| `AZURE_OPENAI_DEPLOYMENT` | Yes (chat deployment) |

## Run

```powershell
streamlit run streamlit_app.py
```

Open the URL Streamlit prints (usually http://localhost:8501).

You can also import `self_critique_agent.py` from another script. The Streamlit app does not reimplement the pipeline.

## Sample behaviour

Ask: `Who directed Inception?`

Expected behaviour: the agent writes a short answer, extracts factual claims, checks Wikipedia, then shows a revised answer. Wikipedia is used only in the verify step.

## Tests

There is no pytest suite in this repository. Live Azure and Wikipedia calls were not run during this export.

## Limitations

- Azure OpenAI is required and may incur cost
- Wikipedia can be empty or unrelated; those claims are marked insufficient
- One revision only; remaining errors are not sent through a second loop

## Attribution

Built by Luyanda Ndaba as academy coursework (AI Day 7 bonus agent). This export keeps the reusable Python agent and Streamlit UI, not grading notebooks or screenshots.
