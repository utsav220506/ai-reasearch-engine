import re
from html import escape
from pathlib import Path
import streamlit as st
from research_brief import generate_research_brief

from answer import answer_question
from comparison import compare_papers
from generation import generate_comparison
from hybrid_retrieval import build_retrievers
from index_corpus import main as index_corpus, make_paper_id


ROOT_DIR = Path(__file__).resolve().parent.parent
PAPERS_DIR = ROOT_DIR / "data" / "papers"
PAPERS_DIR.mkdir(parents=True, exist_ok=True)


st.set_page_config(
    page_title="AI Research Engine",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    .workspace-bar {
        display: flex;
        align-items: center;
        gap: .8rem;
        min-height: 3.3rem;
        padding: .35rem 0 .8rem;
        margin-bottom: .3rem;
    }
    .workspace-mark {
        display: grid;
        place-items: center;
        width: 2rem;
        height: 2rem;
        border: 1px solid rgba(170, 156, 235, .38);
        border-radius: .45rem;
        color: #c6bbf5;
        background: rgba(133, 114, 205, .12);
        font-size: .72rem;
        font-weight: 700;
        letter-spacing: .05em;
    }
    .workspace-name { color: #efedf4; font-size: .94rem; font-weight: 650; }
    .workspace-context { color: #8f8a9b; font-size: .76rem; margin-top: .12rem; }
    .workspace-nav-rule {
        height: 1px;
        background: rgba(190, 181, 220, .13);
        margin: .15rem 0 1.05rem;
    }
    .page-kicker {
        color: #b7a9ed;
        font-size: .72rem;
        font-weight: 650;
        letter-spacing: .12em;
        text-transform: uppercase;
        margin: .1rem 0 .45rem;
    }
    .page-title {
        color: #f0edf5;
        font-size: 1.75rem;
        font-weight: 620;
        letter-spacing: -.035em;
        line-height: 1.18;
        margin: 0 0 .4rem;
    }
    .page-description {
        max-width: 800px;
        color: #a9a5b2;
        font-size: .94rem;
        line-height: 1.55;
        margin-bottom: 1.05rem;
    }
    .section-label {
        color: #c2b7ed;
        font-size: .72rem;
        font-weight: 650;
        letter-spacing: .1em;
        text-transform: uppercase;
        margin: .3rem 0 .55rem;
    }
    .paper-table-head {
        display: grid;
        grid-template-columns: minmax(11rem, 2fr) minmax(6rem, .85fr) minmax(4rem, .55fr) minmax(6rem, .75fr) minmax(6rem, .85fr);
        align-items: center;
        gap: 1rem;
        color: #888493;
        font-size: .69rem;
        font-weight: 650;
        letter-spacing: .09em;
        text-transform: uppercase;
        padding: .4rem .7rem .65rem;
    }
    .paper-row {
        color: #d9d6e0;
        padding: .85rem .7rem;
        border-top: 1px solid rgba(190, 181, 220, .11);
    }
    .paper-row-title { color: #eeebf3; font-size: .91rem; font-weight: 620; line-height: 1.35; }
    .paper-row-subtitle { color: #9b96a5; font-size: .74rem; letter-spacing: .01em; margin-top: .18rem; line-height: 1.35; }
    .paper-row-meta { color: #aaa6b2; font-size: .79rem; }
    .paper-status {
        display: inline-block;
        color: #bdb4e6;
        font-size: .74rem;
        white-space: nowrap;
    }
    .paper-status::before {
        content: "";
        display: inline-block;
        width: .38rem;
        height: .38rem;
        margin: 0 .42rem .04rem 0;
        border-radius: 50%;
        background: #8f7bcf;
    }
    .paper-status-indexed { color: #b9cdbf; }
    .paper-status-indexed::before { background: #83aa91; }
    .paper-status-unchecked { color: #9b96a5; }
    .paper-status-unchecked::before { background: #777381; }
    .workflow-step-wrap {
        border-top: 1px solid rgba(190, 181, 220, .13);
        padding-top: .55rem;
        min-height: 4.2rem;
    }
    .collection-count { color: #e9e6ef; font-size: 1.05rem; font-weight: 620; }
    .collection-count span { color: #9691a0; font-size: .78rem; font-weight: 450; margin-left: .35rem; }
    .workflow-step { color: #bdb8c7; font-size: .78rem; line-height: 1.45; }
    .workflow-step strong { color: #d7d0f0; font-weight: 600; }
    .workflow-number { color: #9184cb; font-size: .68rem; letter-spacing: .1em; margin-bottom: .25rem; }
    .reading-note {
        color: #aaa6b2;
        font-size: .83rem;
        line-height: 1.5;
    }
    .research-rule {
        height: 1px;
        background: rgba(190, 181, 220, .11);
        margin: .1rem 0 .25rem;
    }
    .st-key-brief-reading { max-width: 920px; }
    .st-key-brief-reading h1 { font-size: 1.45rem; line-height: 1.25; margin: .4rem 0 .8rem; }
    .st-key-brief-reading h2 { font-size: 1.12rem; line-height: 1.35; margin: 1.2rem 0 .55rem; }
    .st-key-brief-reading h3 { font-size: .98rem; line-height: 1.4; margin: 1rem 0 .45rem; }
    .st-key-brief-reading p, .st-key-brief-reading li { font-size: .9rem; line-height: 1.6; }
    @media (max-width: 760px) {
        .paper-table-head { display: none; }
        .page-title { font-size: 1.5rem; }
        .workflow-step-wrap { min-height: 0; margin-bottom: .45rem; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading the research engine...")
def get_retrieval_stack():
    return build_retrievers(use_reranker=False)


def format_pages(page_start, page_end):
    if page_start is None and page_end is None:
        return "Page unavailable"

    if page_start is None:
        return f"Page {page_end}"

    if page_end is None or page_start == page_end:
        return f"Page {page_start}"

    return f"Pages {page_start}–{page_end}"


def save_uploaded_pdfs(uploaded_files):
    saved_paths = []
    skipped_files = []

    for uploaded_file in uploaded_files:
        original_name = Path(uploaded_file.name).name

        if not original_name.lower().endswith(".pdf"):
            skipped_files.append(original_name)
            continue

        destination = PAPERS_DIR / original_name

        if destination.exists():
            skipped_files.append(
                f"{original_name} (already exists)"
            )
            continue

        destination.write_bytes(uploaded_file.getvalue())
        saved_paths.append(destination)

    return saved_paths, skipped_files


def render_citations(citations):
    if not citations:
        st.info("No citation metadata was returned.")
        return

    for citation in citations:
        title = citation.get("paper_title", "Unknown paper")
        evidence_id = citation.get("evidence_id", "")
        section = citation.get("section") or "Section unavailable"
        pages = format_pages(
            citation.get("page_start"),
            citation.get("page_end"),
        )
        source_file = citation.get("source_file")
        anchor_id = re.sub(r"[^A-Za-z0-9_-]", "", str(evidence_id))
        st.markdown(
            f'<span id="citation-{anchor_id}"></span>',
            unsafe_allow_html=True,
        )

        with st.expander(
            f"{evidence_id} · {title} · {pages}"
        ):
            st.write(f"**Section:** {section}")
            st.write(
                f"**Paper ID:** {citation.get('paper_id', 'Unknown')}"
            )
            st.write(
                f"**Chunk ID:** {citation.get('chunk_id', 'Unknown')}"
            )

            if source_file:
                st.write(f"**Source file:** {source_file}")


def render_evidence(evidence):
    if not evidence:
        st.info("No evidence passages were returned.")
        return

    for item in evidence:
        evidence_id = item.get("evidence_id", "Evidence")
        title = item.get("paper_title", "Unknown paper")
        section = item.get("section") or "Section unavailable"
        pages = format_pages(
            item.get("page_start"),
            item.get("page_end"),
        )

        with st.expander(
            f"{evidence_id}  ·  {title}  ·  {pages}",
            icon=":material/article:",
        ):
            st.caption(
                f"{section}  ·  Chunk {item.get('chunk_key', 'Unknown')}"
            )
            st.markdown(item.get("text", "No passage text available."))


def render_cited_answer(answer):
    highlighted = re.sub(
        r"\[(E\d+)\]",
        r"[\1](#citation-\1)",
        answer,
    )
    st.markdown(highlighted)


def render_comparison_sections(answer):
    """Render generated comparison headings with a clearer reading hierarchy."""
    first_heading = re.search(r"(?m)^#{1,3}\s+([^\n]+)", answer)
    if not first_heading:
        render_cited_answer(answer)
        return

    if first_heading and answer[:first_heading.start()].strip():
        render_cited_answer(answer[:first_heading.start()].strip())

    matches = list(re.finditer(r"(?m)^#{1,3}\s+([^\n]+)\n?", answer))
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(answer)
        body = answer[match.end():end].strip()
        st.markdown(f"#### {match.group(1).strip()}")
        if body:
            render_cited_answer(body)


with st.container(horizontal=True, horizontal_alignment="center", gap=None):
    with st.container(width=1320):
        pdf_files = sorted(PAPERS_DIR.glob("*.pdf"))

        brand_col, navigation_col = st.columns([0.42, 0.58], vertical_alignment="center")
        with brand_col:
            st.markdown(
                '<div class="workspace-bar">'
                '<div class="workspace-mark">AR</div>'
                '<div><div class="workspace-name">AI Research Engine</div>'
                '<div class="workspace-context">Literature workspace · local corpus</div></div>'
                '</div>',
                unsafe_allow_html=True,
            )
        with navigation_col:
            selected_workspace = st.segmented_control(
                "Workspace navigation",
                options=["Research Q&A", "Paper Comparison", "Paper Library"],
                default="Research Q&A",
                key="workspace_navigation",
                persist_state="session",
                label_visibility="collapsed",
                width="stretch",
            )
        st.markdown('<div class="workspace-nav-rule"></div>', unsafe_allow_html=True)

        if selected_workspace == "Research Q&A":
            st.markdown(
                '<div class="page-kicker">Evidence retrieval · local generation</div>'
                '<div class="page-title">Ask the literature</div>'
                '<div class="page-description">Search across your local research collection and build an answer from traceable passages.</div>',
                unsafe_allow_html=True,
            )
            with st.container(border=True):
                with st.form("research_question_form", border=False):
                    question_col, controls_col = st.columns(
                        [0.76, 0.24],
                        gap="large",
                        vertical_alignment="bottom",
                    )
                    with question_col:
                        question = st.text_area(
                            "Research question",
                            placeholder=(
                                "Ask about methods, results, or evidence across your papers…"
                            ),
                            height=112,
                        )
                    with controls_col:
                        st.markdown(
                            '<div class="section-label">Evidence settings</div>',
                            unsafe_allow_html=True,
                        )
                        top_k = st.slider(
                            "Passages to review",
                            min_value=3,
                            max_value=10,
                            value=5,
                            help="Number of retrieved passages supplied to the answer generator.",
                        )
                        submitted = st.form_submit_button(
                            "Search the literature",
                            type="primary",
                            width="stretch",
                        )

            step_cols = st.columns(3, gap="large")
            for column, number, heading, detail in zip(
                step_cols,
                ("01", "02", "03"),
                ("Ask across papers", "Review passages", "Trace every claim"),
                (
                    "Search the local collection with hybrid retrieval.",
                    "Open the source sections behind each answer.",
                    "Follow evidence IDs to paper and page metadata.",
                ),
            ):
                column.markdown(
                    f'<div class="workflow-step-wrap"><div class="workflow-number">{number}</div>'
                    f'<div class="workflow-step"><strong>{heading}</strong><br>{detail}</div>'
                    '</div>',
                    unsafe_allow_html=True,
                )

            if submitted:
                if not question.strip():
                    st.warning("Enter a research question first.")

                elif not pdf_files:
                    st.warning(
                        "Your library is empty. Upload PDFs in the Paper Library tab."
                    )

                else:
                    try:
                        with st.spinner(
                            "Retrieving evidence and generating your answer..."
                        ):
                            stack = get_retrieval_stack()

                            result = answer_question(
                                question.strip(),
                                *stack,
                                top_k=top_k,
                            )

                        st.markdown(
                            '<div class="section-label">Research synthesis</div>',
                            unsafe_allow_html=True,
                        )
                        render_cited_answer(
                            result.get(
                                "answer",
                                "The engine did not return an answer.",
                            )
                        )

                        st.markdown('<div class="section-label">Source citations</div>', unsafe_allow_html=True)
                        render_citations(result.get("citations", []))

                        st.markdown('<div class="section-label">Retrieved evidence</div>', unsafe_allow_html=True)
                        render_evidence(result.get("evidence", []))

                        with st.expander("Question details"):
                            st.write(result.get("question", question.strip()))

                    except Exception as exc:
                        st.error(
                            "The research request failed. Check that the local "
                            "Qdrant database, retrieval modules, and Ollama model "
                            "are available."
                        )
                        st.exception(exc)
        elif selected_workspace == "Paper Comparison":
            st.markdown(
                '<div class="page-kicker">Cross-paper analysis · source-aware reading</div>'
                '<div class="page-title">Compare studies</div>'
                '<div class="page-description">Read methods, findings, and evidence side by side. Each selected paper is retrieved independently.</div>',
                unsafe_allow_html=True,
            )

            available_papers = {}

            for pdf_file in sorted(PAPERS_DIR.glob("*.pdf")):
                paper_id = make_paper_id(pdf_file)
                available_papers[paper_id] = pdf_file.stem.replace("_", " ")

            if len(available_papers) < 2:
                st.info(
                    "Add at least two research papers to your library "
                    "before using Paper Comparison."
                )

            else:
                with st.form("paper_comparison_form", border=False):
                    paper_col, question_col = st.columns([0.43, 0.57], gap="large")
                    with paper_col:
                        selected_paper_ids = st.multiselect(
                            "Papers to compare",
                            options=list(available_papers.keys()),
                            format_func=lambda paper_id: available_papers[paper_id],
                            max_selections=5,
                            help="Choose two to five papers. Each is retrieved independently.",
                        )
                    with question_col:
                        comparison_question = st.text_area(
                            "Analytical question",
                            placeholder=(
                                "Compare the methods, datasets, evaluation metrics, and reported results…"
                            ),
                            height=105,
                        )

                    settings_col, submit_col = st.columns([0.48, 0.30], vertical_alignment="bottom")
                    with settings_col:
                        comparison_top_k = st.slider(
                            "Evidence passages per paper",
                            min_value=2,
                            max_value=5,
                            value=3,
                        )
                    with submit_col:
                        compare_submitted = st.form_submit_button(
                            "Build comparison",
                            type="primary",
                            width="stretch",
                        )

                if selected_paper_ids:
                    paper_summary = "".join(
                        '<span class="paper-status">'
                        + escape(available_papers[paper_id])
                        + '</span>'
                        for paper_id in selected_paper_ids
                    )
                    st.markdown(
                        '<div class="reading-note">Selected studies &nbsp; '
                        + paper_summary
                        + '</div>',
                        unsafe_allow_html=True,
                    )

                step_cols = st.columns(3, gap="large")
                for column, number, heading, detail in zip(
                    step_cols,
                    ("01", "02", "03"),
                    ("Select studies", "Compare evidence", "Export a brief"),
                    (
                        "Choose two to five papers from your collection.",
                        "Findings remain tied to paper-specific passages.",
                        "Keep the comparison and references in Markdown.",
                    ),
                ):
                    column.markdown(
                        f'<div class="workflow-step-wrap"><div class="workflow-number">{number}</div>'
                        f'<div class="workflow-step"><strong>{heading}</strong><br>{detail}</div>'
                        '</div>',
                        unsafe_allow_html=True,
                    )

                if compare_submitted:
                    if len(selected_paper_ids) < 2:
                        st.warning("Select at least two papers.")

                    elif not comparison_question.strip():
                        st.warning("Enter a comparison question.")

                    else:
                        try:
                            with st.spinner(
                                "Retrieving evidence independently from each paper..."
                            ):
                                stack = get_retrieval_stack()

                                model, client, bm25, chunks, reranker = stack

                                comparison_data = compare_papers(
                                    comparison_question.strip(),
                                    selected_paper_ids,
                                    model,
                                    client,
                                    chunks,
                                    reranker=reranker,
                                    top_k_per_paper=comparison_top_k,
                                )

                            with st.spinner(
                                "Generating an evidence-backed comparison..."
                            ):
                                comparison_answer = generate_comparison(
                                    comparison_question.strip(),
                                    comparison_data["evidence"],
                                )
                                research_brief = generate_research_brief(
                                question=comparison_question.strip(),
                                comparison=comparison_answer,
                                papers=comparison_data["papers"],
                                evidence=comparison_data["evidence"],
                            )

                            result_heading, export_action = st.columns(
                                [0.78, 0.22], vertical_alignment="center"
                            )
                            result_heading.markdown(
                                '<div class="section-label">Comparative reading</div>',
                                unsafe_allow_html=True,
                            )
                            export_action.download_button(
                                label="Export brief",
                                data=research_brief,
                                file_name="research_brief.md",
                                mime="text/markdown",
                                icon=":material/download:",
                                width="content",
                            )
                            render_comparison_sections(comparison_answer)

                            with st.expander("Research brief preview", icon=":material/description:"):
                                with st.container(key="brief-reading"):
                                    st.markdown(research_brief)

                            cited_ids = set(
                                re.findall(
                                    r"\[(E\d+)\]",
                                    comparison_answer,
                                )
                            )

                            citation_metadata = [
                                {
                                    "evidence_id": item["evidence_id"],
                                    "paper_id": item["paper_id"],
                                    "paper_title": item["paper_title"],
                                    "source_file": item["source_file"],
                                    "chunk_id": item["chunk_id"],
                                    "section": item["section"],
                                    "page_start": item["page_start"],
                                    "page_end": item["page_end"],
                                }
                                for item in comparison_data["evidence"]
                                if item["evidence_id"] in cited_ids
                            ]

                            st.markdown('<div class="section-label">Source citations</div>', unsafe_allow_html=True)
                            render_citations(citation_metadata)

                            st.markdown('<div class="section-label">Retrieved evidence</div>', unsafe_allow_html=True)
                            render_evidence(comparison_data["evidence"])

                            with st.expander("Comparison details"):
                                st.write(
                                    f"**Question:** "
                                    f"{comparison_data['question']}"
                                )

                                for paper in comparison_data["papers"]:
                                    st.write(
                                        f"- {paper['paper_title']}: "
                                        f"{paper['retrieved_count']} passages"
                                    )

                        except Exception as exc:
                            st.error(
                                "The paper comparison failed. Check that "
                                "Qdrant, the retrieval stack, and Ollama "
                                "are available."
                            )
                            st.exception(exc)


        elif selected_workspace == "Paper Library":
            title_col, actions_col = st.columns([0.76, 0.24], vertical_alignment="center")
            with title_col:
                st.markdown(
                    '<div class="page-kicker">Personal corpus · local PDF collection</div>'
                    '<div class="page-title">Paper library</div>'
                    '<div class="page-description">A focused collection of source documents for retrieval, comparison, and citation.</div>',
                    unsafe_allow_html=True,
                )
            with actions_col:
                with st.popover("Add papers", icon=":material/upload_file:", width="stretch"):
                    uploaded_files = st.file_uploader(
                        "Choose research PDFs",
                        type=["pdf"],
                        accept_multiple_files=True,
                        help="PDFs are stored in your local paper library.",
                        key="library_pdf_upload",
                    )
                    upload_clicked = st.button(
                        "Upload and index",
                        type="primary",
                        width="stretch",
                        disabled=not uploaded_files,
                        key="library_upload_submit",
                    )

            if upload_clicked:
                try:
                    saved_paths, skipped_files = save_uploaded_pdfs(uploaded_files or [])
                    if skipped_files:
                        st.warning("Skipped: " + ", ".join(skipped_files))
                    if saved_paths:
                        try:
                            with st.spinner("Rebuilding the corpus index…"):
                                get_retrieval_stack.clear()
                                index_corpus()
                                get_retrieval_stack.clear()
                        except Exception:
                            for path in saved_paths:
                                path.unlink(missing_ok=True)
                            try:
                                get_retrieval_stack.clear()
                                if list(PAPERS_DIR.glob("*.pdf")):
                                    index_corpus()
                            finally:
                                get_retrieval_stack.clear()
                            st.session_state["library_index_state"] = "Indexing failed; upload was rolled back."
                            raise

                        indexed_names = set(st.session_state.get("indexed_paper_names", []))
                        indexed_names.update(path.name for path in saved_paths)
                        st.session_state["indexed_paper_names"] = sorted(indexed_names)
                        st.session_state["library_index_state"] = (
                            f"Index ready · {len(saved_paths)} paper(s) added"
                        )
                        st.rerun()
                    elif not skipped_files:
                        st.info("Choose at least one PDF to upload.")
                except Exception as exc:
                    st.error("Indexing failed. The new upload was rolled back where possible.")
                    st.exception(exc)

            pdf_files = sorted(PAPERS_DIR.glob("*.pdf"))
            indexed_names = set(st.session_state.get("indexed_paper_names", []))
            state_message = st.session_state.get("library_index_state")
            if state_message:
                st.success(state_message)

            count_col, note_col = st.columns([0.23, 0.77], vertical_alignment="center")
            count_col.markdown(
                f'<div class="collection-count">{len(pdf_files)}<span>papers in collection</span></div>',
                unsafe_allow_html=True,
            )
            note_col.markdown(
                '<div class="reading-note">Files remain on this device. Existing files are shown as '
                '“Index status not checked” until a successful indexing run is recorded in this session.</div>',
                unsafe_allow_html=True,
            )

            if pdf_files:
                st.markdown(
                    '<div class="paper-table-head"><span>Paper · source file</span>'
                    '<span>Paper identifier</span><span>File size</span><span>Index state</span><span>Actions</span></div>',
                    unsafe_allow_html=True,
                )
                for pdf_file in pdf_files:
                    paper_title = escape(pdf_file.stem.replace("_", " "))
                    source_name = escape(pdf_file.name)
                    paper_id = escape(make_paper_id(pdf_file))
                    size_mb = pdf_file.stat().st_size / (1024 * 1024)
                    index_state = (
                        "Indexed" if pdf_file.name in indexed_names
                        else "Not checked"
                    )
                    cells = st.columns([2.0, 0.85, 0.55, 0.75, 0.85], vertical_alignment="center")
                    cells[0].markdown(
                        f'<div class="paper-row-title">{paper_title}</div>'
                        f'<div class="paper-row-subtitle">{source_name}</div>',
                        unsafe_allow_html=True,
                    )
                    cells[1].markdown(f'<div class="paper-row-meta">{paper_id}</div>', unsafe_allow_html=True)
                    cells[2].markdown(f'<div class="paper-row-meta">{size_mb:.1f} MB</div>', unsafe_allow_html=True)
                    cells[3].markdown(
                        f'<span class="paper-status {"paper-status-indexed" if index_state == "Indexed" else "paper-status-unchecked"}">{escape(index_state)}</span>',
                        unsafe_allow_html=True,
                    )
                    with cells[4].popover(
                        "View",
                        icon=":material/more_horiz:",
                        key=f"paper_details_{paper_id}",
                    ):
                        st.write(f"**Title:** {pdf_file.stem.replace('_', ' ')}")
                        st.write(f"**Paper ID:** `{make_paper_id(pdf_file)}`")
                        st.write(f"**Source:** `{pdf_file.name}`")
                        st.write(f"**Index state:** {index_state}")
                    st.markdown('<div class="research-rule"></div>', unsafe_allow_html=True)
            else:
                st.info("Your local literature collection is empty. Add PDFs to begin building the corpus.")
