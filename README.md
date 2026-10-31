# Resume Matching Agent

A standalone conversational candidate screening project built with LangGraph and local resume files. The corpus contains 100 synthetic resumes.

## Setup

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Run the interactive chat

```powershell
.\.venv\Scripts\python.exe matching_agent.py
```

Try: “Find me candidates with React and 3+ years experience”, “Compare the top 3 matches side by side”, “Why did Jane Smith rank higher than Marcus Johnson?”, and then refine criteria in the same conversation.

Run `Run second round deep analysis` to inspect the top ten with supporting resume evidence. Run `Run final round hire/no-hire recommendation` to produce a documented evidence-threshold recommendation. The first round scans all 100 resumes and presents a top-ten shortlist.

Run one request with:

```powershell
.\.venv\Scripts\python.exe matching_agent.py --once "Find me candidates with React and 3+ years experience"
```

## Assignment artifacts

- `matching_agent.py`: LangGraph pipeline, persistent state, local TF-IDF RAG retrieval, three screening rounds, rankings, interview questions, and report export.
- `STATE_MACHINE.md`: workflow diagram.
- `TEST_SCENARIOS.md`: six conversation flows.
- `resumes/`: ten dummy candidate resumes.
- `tests/`: PDF/DOCX file reading and agent behavior coverage.

Run tests with `.\.venv\Scripts\python.exe -m pytest -q`.

