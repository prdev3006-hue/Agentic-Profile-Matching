"""Explainable resume matching agent with a LangGraph workflow and CLI.

The graph uses LangGraph when installed (see requirements.txt); its node
functions remain directly callable for deterministic, offline demonstrations.
"""
from __future__ import annotations

import argparse
import json
import re
import math
from collections import Counter
from pathlib import Path
from typing import Any, TypedDict

from fs_tools import list_files, read_file, write_file

try:
    from langgraph.graph import END, START, StateGraph
except ImportError:  # Allows file tools and unit tests in minimal environments.
    END = START = StateGraph = None  # type: ignore[assignment,misc]


class MatchState(TypedDict, total=False):
    conversation_history: list[dict[str, str]]
    user_message: str
    job_description: str
    requirements: dict[str, list[str]]
    candidates: list[dict[str, Any]]
    shortlist: list[dict[str, Any]]
    ranking_reasoning: list[str]
    report: str
    feedback: str
    round: int
    rag_results: list[dict[str, Any]]
    final_recommendation: str


SKILL_TERMS = {
    "react": ["react"], "python": ["python"], "javascript": ["javascript", " js "],
    "typescript": ["typescript"], "node": ["node.js", "nodejs"], "sql": ["sql", "postgres", "mysql"],
    "aws": ["aws", "amazon web services"], "docker": ["docker"], "kubernetes": ["kubernetes", " k8s"],
    "java": ["java"], "c#": ["c#", "c sharp"], "communication": ["communication", "stakeholder"],
}


def extract_requirements(jd: str) -> dict[str, list[str]]:
    """Extract skill must-haves and preferences from natural language JD text."""
    lower = f" {jd.lower()} "
    must, nice = [], []
    for skill, aliases in SKILL_TERMS.items():
        if any(alias in lower for alias in aliases):
            # A nearby requirement cue classifies explicit essentials.
            positions = [lower.find(alias) for alias in aliases if lower.find(alias) >= 0]
            pos = min(positions)
            context = lower[max(0, pos - 60):pos]
            target = must if re.search(r"(must|required|mandatory|minimum|need|\b3\+ years\b)[^.;\n]{0,45}$", context) else nice
            target.append(skill)
    min_years = 0
    for match in re.finditer(r"(\d+)\s*\+?\s*years?", lower):
        min_years = max(min_years, int(match.group(1)))
    return {"must_have": sorted(set(must)), "nice_to_have": sorted(set(nice)), "min_years": min_years}


def compare_candidates(candidate_ids: list[str], candidates: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """Return side-by-side evidence for candidates by id/name."""
    rows = candidates or []
    selected = [c for c in rows if c.get("id") in candidate_ids or c.get("name") in candidate_ids]
    return {"candidates": [{k: c.get(k) for k in ("id", "name", "score", "matched_skills", "gaps", "years", "reasoning")} for c in selected]}


def generate_interview_questions(candidate_id: str, candidates: list[dict[str, Any]] | None = None) -> list[str]:
    candidate = next((c for c in candidates or [] if c.get("id") == candidate_id or c.get("name", "").lower() == candidate_id.lower()), {})
    gaps = candidate.get("gaps", [])
    questions = [f"Walk me through a project where you used {skill} and how you measured its impact." for skill in candidate.get("matched_skills", [])[:2]]
    questions += [f"What is your practical experience with {gap}?" for gap in gaps[:2]]
    return (questions or ["Describe a recent project most relevant to this role.", "What would you focus on in your first 90 days?"])[:4]


def _parse_turn(text: str, state: MatchState) -> None:
    lower = text.lower()
    # New JD/initial request versus adjustment to the active criteria.
    if any(marker in lower for marker in ("job description:", "role:", "hiring for", "find me", "find candidates", "looking for")):
        state["job_description"] = text
        state["requirements"] = extract_requirements(text)
    elif state.get("job_description") and any(word in lower for word in ("require", "prefer", "must", "remove", "add", "at least", "years", "react", "python", "sql", "aws", "docker")):
        state["job_description"] += "\nAdditional criteria: " + text
        state["requirements"] = extract_requirements(state["job_description"])
    elif not state.get("job_description"):
        state["job_description"] = text
        state["requirements"] = extract_requirements(text)


def _experience_years(text: str) -> int:
    matches = [int(n) for n in re.findall(r"(\d+)\s*\+?\s*years?", text.lower())]
    return max(matches, default=0)


def rag_search(query: str, directory: str = "resumes", top_k: int = 100) -> list[dict[str, Any]]:
    """Local TF-IDF retrieval over resume text; returns evidence snippets and scores."""
    docs = []
    for file in list_files(directory):
        if not file.get("success") or file.get("extension") not in (".txt", ".pdf", ".docx") or "summary" in file["name"].lower():
            continue
        result = read_file(file["path"])
        if result.get("success"):
            docs.append((file, result["content"]))
    tokens = lambda value: re.findall(r"[a-z0-9+#.]+", value.lower())
    query_terms = set(tokens(query))
    if not query_terms:
        return []
    doc_terms = [Counter(tokens(text)) for _, text in docs]
    df = Counter(term for terms in doc_terms for term in terms)
    scored = []
    for (file, content), terms in zip(docs, doc_terms):
        score = sum((1 + math.log(terms[t])) * (math.log(1 + len(docs) / (1 + df[t]))) for t in query_terms if terms[t])
        if score:
            # Return compact evidence around first matching query term.
            low = content.lower()
            positions = [low.find(term) for term in query_terms if low.find(term) >= 0]
            pos = min(positions) if positions else 0
            snippet = content[max(0, pos - 100):pos + 220].replace("\n", " ").strip()
            scored.append({"id": file["name"], "path": file["path"], "score": round(score, 4), "snippet": snippet})
    return sorted(scored, key=lambda row: row["score"], reverse=True)[:top_k]


def _search(state: MatchState, directory: str = "resumes") -> list[dict[str, Any]]:
    results = []
    for file in list_files(directory):
        if not file.get("success") or file.get("extension") not in (".txt", ".pdf", ".docx"):
            continue
        if "summary" in file["name"].lower():
            continue
        read = read_file(file["path"])
        if not read.get("success"):
            continue
        content = read["content"]
        first_line = next((line.strip() for line in content.splitlines() if line.strip()), "")
        filename_name = re.sub(r"\.[^.]+$", "", file["name"]).replace("resume_", "").replace("_", " ").title()
        name = first_line if len(first_line) < 60 and not re.search(r"\b(experience|engineer|developer|skills|education)\b", first_line, re.I) else filename_name
        results.append({"id": file["name"], "name": name, "path": file["path"], "text": content, "years": _experience_years(content)})
    return results


def _rank(state: MatchState) -> list[dict[str, Any]]:
    req = state.get("requirements", {})
    ranked = []
    for candidate in state.get("candidates", []):
        content = candidate["text"].lower()
        matched = [s for s in req.get("must_have", []) + req.get("nice_to_have", []) if any(a in content for a in SKILL_TERMS.get(s, [s]))]
        missing = [s for s in req.get("must_have", []) if s not in matched]
        years_ok = candidate["years"] >= req.get("min_years", 0)
        # Must-have misses and minimum-years misses carry the largest penalty.
        denominator = max(1, len(req.get("must_have", [])) * 2 + len(req.get("nice_to_have", [])) + bool(req.get("min_years")))
        points = (2 * sum(s in matched for s in req.get("must_have", [])) + sum(s in matched for s in req.get("nice_to_have", [])) + int(years_ok and bool(req.get("min_years"))))
        score = round(100 * points / denominator)
        strengths = matched or ["Relevant experience described in resume"]
        gaps = missing + ([f"less than {req['min_years']} years stated" ] if req.get("min_years") and not years_ok else [])
        candidate.update(score=score, matched_skills=matched, gaps=gaps, strengths=strengths, reasoning=f"{len(matched)} requested skill(s) evidenced; {len(gaps)} gap(s); {candidate['years']} years explicitly stated.")
        ranked.append(candidate)
    return sorted(ranked, key=lambda c: (c["score"], c["years"], c["name"]), reverse=True)


def _report(state: MatchState) -> str:
    ranked = state.get("shortlist", [])
    lines = ["Candidate Match Report", f"Requirements: {json.dumps(state.get('requirements', {}), ensure_ascii=False)}", ""]
    if not ranked:
        lines.append("No readable resumes found.")
    rag_results = state.get("rag_results", [])
    if rag_results:
        lines.append(f"Local RAG retrieved {len(rag_results)} relevant resume(s); top evidence source: {rag_results[0]['id']} (TF-IDF {rag_results[0]['score']}).")
    for i, c in enumerate(ranked, 1):
        lines += [f"{i}. {c['name']} - {c['score']}% match", f"   Strengths: {', '.join(c['strengths'])}", f"   Gaps: {', '.join(c['gaps']) or 'No stated must-have gaps'}", f"   Reason: {c['reasoning']}"]
        if 45 <= c["score"] < 70:
            lines.append("   Improvement: add concrete project evidence and measurable outcomes for the missing criteria.")
        if state.get("round", 1) >= 2:
            lines.append(f"   Deep review evidence: {c.get('evidence', 'No additional matching evidence found.')}")
    lines += [""]
    if state.get("round", 1) >= 3:
        lines.append("Final round recommendation: " + state.get("final_recommendation", "No recommendation available."))
    else:
        lines.append("Recommendation: advance the highest-ranked candidate(s) to a structured interview; validate resume claims before a hiring decision.")
    return "\n".join(lines)


def build_graph(directory: str = "resumes"):
    """Build START→parse→extract→search→rank→report→feedback→END graph."""
    if StateGraph is None:
        return None
    def parse(state: MatchState):
        _parse_turn(state.get("user_message", ""), state)
        return state
    def extract(state: MatchState):
        state["requirements"] = extract_requirements(state.get("job_description", ""))
        return state
    def search(state: MatchState):
        state["candidates"] = _search(state, directory)
        state["rag_results"] = rag_search(state.get("job_description", ""), directory, 100)
        return state
    def rank(state: MatchState):
        ranked = _rank(state)
        retrieval_scores = {row["id"]: row["score"] for row in state.get("rag_results", [])}
        for candidate in ranked:
            candidate["retrieval_score"] = retrieval_scores.get(candidate["id"], 0.0)
        state["shortlist"] = ranked[:10]
        state["ranking_reasoning"] = [f"{c['name']}: {c['reasoning']}" for c in state["shortlist"]]
        return state
    def second_round(state: MatchState):
        if state.get("round", 1) >= 2:
            terms = state.get("requirements", {}).get("must_have", []) + state.get("requirements", {}).get("nice_to_have", [])
            for candidate in state.get("shortlist", []):
                evidence = [line.strip() for line in candidate["text"].splitlines() if any(term.lower() in line.lower() for term in terms)]
                candidate["evidence"] = " | ".join(evidence) or "No direct skill evidence found."
        return state
    def final_round(state: MatchState):
        if state.get("round", 1) >= 3:
            qualified = [c for c in _rank(state) if c["score"] >= 70 and not c["gaps"]]
            state["final_recommendation"] = (f"HIRE recommendation for {qualified[0]['name']} ({qualified[0]['score']}% evidence match), subject to interviews and reference checks." if qualified else "NO-HIRE recommendation at this stage: no candidate met the evidence threshold (70%) with all stated must-haves. Revisit criteria or gather more evidence.")
        return state
    def report(state: MatchState):
        state["report"] = _report(state)
        return state
    def feedback(state: MatchState):
        state["conversation_history"] = state.get("conversation_history", []) + [{"role": "user", "content": state.get("user_message", "")}, {"role": "assistant", "content": state.get("report", "")}]
        return state
    graph = StateGraph(MatchState)
    for name, node in (("parse_jd", parse), ("extract_requirements", extract), ("search_resumes", search), ("rank_candidates", rank), ("second_round_analysis", second_round), ("final_round_recommendation", final_round), ("generate_report", report), ("human_feedback", feedback)):
        graph.add_node(name, node)
    graph.add_edge(START, "parse_jd")
    for a, b in zip(("parse_jd", "extract_requirements", "search_resumes", "rank_candidates", "second_round_analysis", "final_round_recommendation", "generate_report", "human_feedback"), ("extract_requirements", "search_resumes", "rank_candidates", "second_round_analysis", "final_round_recommendation", "generate_report", "human_feedback", END)):
        graph.add_edge(a, b)
    return graph.compile()


class MatchingAgent:
    def __init__(self, directory: str = "resumes"):
        self.directory = directory
        self.state: MatchState = {"conversation_history": [], "requirements": {}, "candidates": [], "shortlist": [], "round": 1}
        self.graph = build_graph(directory)

    def chat(self, message: str) -> str:
        lower_message = message.lower()
        if "final round" in lower_message or "final recommendation" in lower_message or "hire/no-hire" in lower_message:
            self.state["round"] = 3
        elif "second round" in lower_message or "deep analysis" in lower_message or "deep review" in lower_message:
            self.state["round"] = 2
        elif not self.state.get("job_description"):
            self.state["round"] = 1
        self.state["user_message"] = message
        if self.graph:
            self.state = self.graph.invoke(self.state)  # type: ignore[assignment]
        else:
            _parse_turn(message, self.state)
            self.state["requirements"] = extract_requirements(self.state.get("job_description", ""))
            self.state["candidates"] = _search(self.state, self.directory)
            self.state["rag_results"] = rag_search(self.state.get("job_description", ""), self.directory, 100)
            ranked = _rank(self.state)
            if self.state.get("round", 1) >= 2:
                for candidate in ranked[:10]:
                    evidence_lines = [line.strip() for line in candidate["text"].splitlines() if any(term.lower() in line.lower() for term in self.state["requirements"].get("must_have", []) + self.state["requirements"].get("nice_to_have", []))]
                    candidate["evidence"] = " | ".join(evidence_lines) or "No direct skill evidence found."
            self.state["shortlist"] = ranked[:10]
            retrieval_scores = {row["id"]: row["score"] for row in self.state.get("rag_results", [])}
            for candidate in self.state["shortlist"]:
                candidate["retrieval_score"] = retrieval_scores.get(candidate["id"], 0.0)
            if self.state.get("round", 1) >= 3:
                qualified = [c for c in ranked if c["score"] >= 70 and not c["gaps"]]
                self.state["final_recommendation"] = f"HIRE recommendation for {qualified[0]['name']} ({qualified[0]['score']}% evidence match), subject to interviews and reference checks." if qualified else "NO-HIRE recommendation at this stage: no candidate met the evidence threshold (70%) with all stated must-haves. Revisit criteria or gather more evidence."
            self.state["ranking_reasoning"] = [c["reasoning"] for c in self.state["shortlist"]]
            self.state["report"] = _report(self.state)
            self.state["conversation_history"] += [{"role": "user", "content": message}, {"role": "assistant", "content": self.state["report"]}]
        return self._answer(message)

    def _answer(self, message: str) -> str:
        lower = message.lower()
        ranked = self.state.get("shortlist", [])
        if self.state.get("round", 1) >= 3:
            return self.state.get("report", "")
        if self.state.get("round", 1) == 2:
            return self.state.get("report", "")
        if "compare" in lower:
            return json.dumps(compare_candidates([c["id"] for c in ranked[:3]], ranked), indent=2)
        if "question" in lower or "interview" in lower:
            name = next((c["name"] for c in ranked if c["name"].lower() in lower), ranked[0]["id"] if ranked else "")
            return "\n".join(generate_interview_questions(name, ranked)) if name else "No candidates available yet."
        if "why" in lower and len(ranked) > 1:
            first, second = ranked[:2]
            return f"{first['name']} ranked above {second['name']} because {first['reasoning']} {second['name']}: {second['reasoning']}"
        if any(x in lower for x in ("save", "export", "write report")):
            result = write_file("outputs/candidate_match_report.txt", self.state.get("report", ""))
            return f"{self.state['report']}\n\nSaved: {result.get('metadata', {}).get('path', result.get('error'))}"
        return self.state.get("report", "")


def main() -> None:
    parser = argparse.ArgumentParser(description="Conversational resume matching agent")
    parser.add_argument("--resumes", default="resumes", help="Resume directory")
    parser.add_argument("--once", help="Run one request and exit")
    args = parser.parse_args()
    agent = MatchingAgent(args.resumes)
    if args.once:
        print(agent.chat(args.once))
        return
    print("Resume Matching Agent (type 'quit' to exit). Describe a role or refine criteria.")
    while True:
        try:
            message = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if message.lower() in {"quit", "exit"}:
            break
        if message:
            print(f"\nAgent: {agent.chat(message)}\n")


if __name__ == "__main__":
    main()
