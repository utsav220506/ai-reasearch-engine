
import re
from collections import defaultdict
from typing import Any


def _format_pages(page_start, page_end):
    if page_start is None and page_end is None:
        return "Pages unavailable"

    if page_start is None:
        return f"Page {page_end}"

    if page_end is None or page_start == page_end:
        return f"Page {page_start}"

    return f"Pages {page_start}–{page_end}"


def _clean_text(value):
    if not isinstance(value, str):
        return ""

    return value.strip()


def _table_cell(value):
    """Escape Markdown table syntax, including multiline metadata."""
    value = _clean_text(value)
    return (value.replace("\\", "\\\\").replace("|", "\\|")
            .replace("\r\n", "<br>").replace("\n", "<br>")
            .replace("\r", "<br>"))


def generate_research_brief(
    question: str,
    comparison: str,
    papers: list[dict[str, Any]],
    evidence: list[dict[str, Any]],
) -> str:

    if not isinstance(question, str) or not question.strip():
        raise ValueError("A non-empty research question is required.")

    if not isinstance(comparison, str) or not comparison.strip():
        raise ValueError("A comparison result is required.")

    if not isinstance(papers, list) or not isinstance(evidence, list) or not evidence:
        raise ValueError("No retrieved evidence is available.")

    evidence_ids = [item.get("evidence_id") for item in evidence
                    if isinstance(item, dict)]
    if len(evidence_ids) != len(evidence) or any(not isinstance(i, str) or not i for i in evidence_ids):
        raise ValueError("Every evidence item must have a valid evidence ID.")
    if len(set(evidence_ids)) != len(evidence_ids):
        raise ValueError("Evidence IDs must be unique.")

    cited_ids = set(
        re.findall(r"\[(E\d+)\]", comparison)
    )

    evidence_by_paper = defaultdict(list)

    for item in evidence:
        evidence_by_paper[item.get("paper_id", "unknown")].append(item)

    lines = [
        "# Research Brief",
        "",
        "## Research Question",
        "",
        question.strip(),
        "",
        "## Executive Summary",
        "",
        comparison.strip(),
        "",
        "## Papers Included",
        "",
    ]

    for paper in papers:
        paper_id = paper.get("paper_id", "unknown")
        title = paper.get("paper_title", paper_id)
        retrieved_count = paper.get("retrieved_count", 0)

        lines.extend([
            f"### {title}",
            "",
            f"- **Paper ID:** `{paper_id}`",
            f"- **Retrieved evidence passages:** {retrieved_count}",
            "",
        ])

    lines.extend([
        "## Evidence by Paper",
        "",
        "The following profiles summarize only the passages retrieved "
        "for this comparison. They are not exhaustive descriptions "
        "of the full papers.",
        "",
    ])

    for paper_id, items in evidence_by_paper.items():
        title = items[0].get("paper_title", paper_id)

        lines.extend([
            f"### {title}",
            "",
        ])

        for item in items:
            evidence_id = item.get("evidence_id", "Unknown")
            section = item.get("section") or "Section unavailable"
            pages = _format_pages(
                item.get("page_start"),
                item.get("page_end"),
            )
            passage = _clean_text(item.get("text", ""))

            lines.extend([
                f"**[{evidence_id}] {section} — {pages}**",
                "",
                passage,
                "",
            ])

    lines.extend([
        "## Comparative Analysis",
        "",
        comparison.strip(),
        "",
        "## Evidence Matrix",
        "",
        "| Evidence ID | Paper | Section | Pages | Cited in comparison |",
        "|---|---|---|---|---|",
    ])

    for item in evidence:
        evidence_id = item.get("evidence_id", "Unknown")
        title = item.get("paper_title", "Unknown paper")
        section = item.get("section") or "Unavailable"
        pages = _format_pages(
            item.get("page_start"),
            item.get("page_end"),
        )
        cited = "Yes" if evidence_id in cited_ids else "No"

        lines.append(
            f"| {_table_cell(evidence_id)} | {_table_cell(title)} | {_table_cell(section)} | "
            f"{pages} | {cited} |"
        )

    lines.extend([
        "",
        "## Evidence Gaps and Interpretation",
        "",
        "This brief is based on retrieved passages, not a complete "
        "independent review of every page in each paper. An item not "
        "represented in the retrieved passages should be treated as "
        "not established by this evidence, rather than proof that "
        "the full paper does not contain it.",
        "",
        "## References",
        "",
    ])

    seen_references = set()

    for item in evidence:
        paper_id = item.get("paper_id", "unknown")

        if paper_id in seen_references:
            continue

        seen_references.add(paper_id)

        title = item.get("paper_title", "Unknown paper")
        source_file = item.get("source_file")

        lines.append(f"- **{title}**")

        if source_file:
            lines.append(f"  - Source file: `{source_file}`")

        lines.append(f"  - Paper ID: `{paper_id}`")

    lines.extend([
        "",
        "---",
        "",
        "*Generated by AI Research Engine. This brief preserves "
        "the comparison output and retrieved evidence metadata. "
        "It does not independently verify claims against the "
        "complete source documents.*",
        "",
    ])

    return "\n".join(lines)
