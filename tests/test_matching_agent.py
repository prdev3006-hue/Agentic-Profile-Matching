from pathlib import Path

from docx import Document
from pypdf import PdfReader, PdfWriter

from fs_tools import read_file
from matching_agent import MatchingAgent, compare_candidates, extract_requirements, generate_interview_questions, rag_search


def test_extract_and_refine_requirements():
    req = extract_requirements("Must have React and at least 3 years; SQL is nice to have")
    assert "react" in req["must_have"]
    assert "sql" in req["nice_to_have"]
    assert req["min_years"] == 3


def test_agent_ranks_and_refines(tmp_path: Path):
    directory = tmp_path / "resumes"
    directory.mkdir()
    (directory / "resume_a.txt").write_text("Alex\nReact engineer, 5 years experience", encoding="utf-8")
    (directory / "resume_b.txt").write_text("Blair\nReact engineer, 1 year experience", encoding="utf-8")
    agent = MatchingAgent(str(directory))
    report = agent.chat("Find me candidates with React and 3+ years experience")
    assert "Alex" in report
    assert agent.state["shortlist"][0]["name"] == "Alex"
    agent.chat("Make 6 years required")
    assert agent.state["requirements"]["min_years"] == 6


def test_screening_rounds_and_rag(tmp_path: Path):
    directory = tmp_path / "resumes"
    directory.mkdir()
    for i in range(12):
        (directory / f"resume_{i}.txt").write_text(f"Person {i}\nReact Developer\nExperience: {i + 1} years\nSkills: React, SQL", encoding="utf-8")
    agent = MatchingAgent(str(directory))
    agent.chat("Find me React candidates with 3+ years experience")
    assert len(agent.state["shortlist"]) == 10
    assert rag_search("React 3 years", str(directory), 5)
    agent.chat("Run second round deep analysis")
    assert agent.state["round"] == 2
    assert all("evidence" in c for c in agent.state["shortlist"])
    report = agent.chat("Run final round hire/no-hire recommendation")
    assert agent.state["round"] == 3
    assert "recommendation" in report.lower()


def test_candidate_tools():
    candidates = [{"id": "a", "name": "Alex", "score": 90, "matched_skills": ["react"], "gaps": [], "years": 5, "reasoning": "good"}]
    assert compare_candidates(["a"], candidates)["candidates"][0]["name"] == "Alex"
    assert generate_interview_questions("a", candidates)


def test_read_pdf_and_docx(tmp_path: Path):
    pdf_path = tmp_path / "resume.pdf"
    writer = PdfWriter()
    writer.add_blank_page(width=300, height=300)
    writer.add_metadata({"/Title": "PDF resume"})
    with pdf_path.open("wb") as handle:
        writer.write(handle)
    # The generated PDF has a valid page even when no text layer exists.
    result = read_file(str(pdf_path))
    assert result["success"] is True
    assert PdfReader(str(pdf_path)).get_num_pages() == 1

    docx_path = tmp_path / "resume.docx"
    doc = Document()
    doc.add_paragraph("Taylor Doe — React Developer")
    doc.add_table(rows=1, cols=2).rows[0].cells[0].text = "5 years experience"
    doc.save(docx_path)
    result = read_file(str(docx_path))
    assert result["success"] is True
    assert "React Developer" in result["content"]
    assert "5 years experience" in result["content"]
