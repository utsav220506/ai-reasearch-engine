
from research_brief import generate_research_brief


def sample_data():
    papers = [
        {
            "paper_id": "paper-1",
            "paper_title": "Paper One",
            "retrieved_count": 1,
        },
        {
            "paper_id": "paper-2",
            "paper_title": "Paper Two",
            "retrieved_count": 1,
        },
    ]

    evidence = [
        {
            "evidence_id": "E1",
            "paper_id": "paper-1",
            "paper_title": "Paper One",
            "source_file": "paper_one.pdf",
            "section": "Methods",
            "page_start": 2,
            "page_end": 3,
            "text": "Paper One uses a Random Forest model.",
        },
        {
            "evidence_id": "E2",
            "paper_id": "paper-2",
            "paper_title": "Paper Two",
            "source_file": "paper_two.pdf",
            "section": "Results",
            "page_start": 5,
            "page_end": 5,
            "text": "Paper Two evaluates an LSTM model.",
        },
    ]

    return papers, evidence


def test_brief_contains_comparison_and_evidence():
    papers, evidence = sample_data()

    brief = generate_research_brief(
        question="Compare the models.",
        comparison="## Findings\n\nPaper One uses RF [E1].",
        papers=papers,
        evidence=evidence,
    )

    assert "# Research Brief" in brief
    assert "Compare the models." in brief
    assert "Paper One uses RF [E1]." in brief
    assert "Random Forest model" in brief
    assert "LSTM model" in brief


def test_brief_marks_cited_and_uncited_evidence():
    papers, evidence = sample_data()

    brief = generate_research_brief(
        question="Compare the models.",
        comparison="Paper One uses RF [E1].",
        papers=papers,
        evidence=evidence,
    )

    assert "| E1 | Paper One | Methods | Pages 2–3 | Yes |" in brief
    assert "| E2 | Paper Two | Results | Page 5 | No |" in brief


def test_brief_rejects_missing_evidence():
    try:
        generate_research_brief(
            question="Compare the models.",
            comparison="Some comparison.",
            papers=[],
            evidence=[],
        )
    except ValueError as error:
        assert "No retrieved evidence" in str(error)
    else:
        raise AssertionError("Expected ValueError")


def test_brief_rejects_duplicate_evidence_ids():
    papers, evidence = sample_data()
    evidence[1]["evidence_id"] = "E1"
    try:
        generate_research_brief("Question", "Summary", papers, evidence)
    except ValueError as error:
        assert "unique" in str(error)
    else:
        raise AssertionError("Expected duplicate evidence IDs to be rejected")


def test_markdown_table_metadata_is_single_line_and_escaped():
    papers, evidence = sample_data()
    evidence[0]["paper_title"] = "One | Two\\Three\nFour"
    brief = generate_research_brief("Question", "Summary", papers, evidence)
    assert "One \\| Two\\\\Three<br>Four" in brief
