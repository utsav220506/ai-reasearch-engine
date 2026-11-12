import re
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
    initial_sidebar_state="expanded",
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
            f"{evidence_id} · {title} · {pages}"
        ):
            st.write(f"**Section:** {section}")
            st.write(
                f"**Chunk:** {item.get('chunk_key', 'Unknown')}"
            )
            st.write(item.get("text", "No passage text available."))


st.title("🔬 AI Research Engine")
st.caption(
    "A local research assistant for retrieving, comparing, "
    "and answering questions across scientific papers."
)

with st.sidebar:
    st.header("Research Workspace")

    st.write("**Retrieval:** Dense + BM25 + RRF")
    st.write("**Generation:** Local Ollama model")
    st.write("**Vector database:** Local Qdrant")
    st.write("**Embeddings:** BGE-base-en-v1.5")

    st.divider()

    st.subheader("Corpus Overview")

    pdf_files = sorted(PAPERS_DIR.glob("*.pdf"))

    st.metric("PDF papers", len(pdf_files))

    if pdf_files:
        with st.expander("View indexed corpus files"):
            for pdf_file in pdf_files:
                st.write(f"📄 {pdf_file.name}")
    else:
        st.warning("No PDFs found. Upload papers in the Library tab.")

    st.divider()

    st.caption("Designed for local execution on your machine.")


research_tab, comparison_tab, library_tab = st.tabs(
    [
        "💬 Research Q&A",
        "📊 Paper Comparison",
        "📚 Paper Library",
    ]
)


with research_tab:
    st.subheader("Ask your research corpus")

    st.write(
        "Ask a question about the papers in your library. "
        "Answers include evidence passages and source metadata."
    )

    with st.form("research_question_form"):
        question = st.text_area(
            "Research question",
            placeholder=(
                "Example: Which machine learning models were used, "
                "and how did their results compare?"
            ),
            height=110,
        )

        top_k = st.slider(
            "Evidence passages",
            min_value=3,
            max_value=10,
            value=5,
            help="Number of retrieved passages supplied to the answer generator.",
        )

        submitted = st.form_submit_button(
            "Research",
            type="primary",
            use_container_width=True,
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

                st.divider()
                st.subheader("Answer")
                st.markdown(
                    result.get(
                        "answer",
                        "The engine did not return an answer.",
                    )
                )

                st.subheader("Source Citations")
                render_citations(result.get("citations", []))

                st.subheader("Retrieved Evidence")
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
with comparison_tab:
    st.subheader("Compare research papers")

    st.write(
        "Select multiple papers to compare their methods, "
        "results, findings, and limitations using retrieved evidence."
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
        with st.form("paper_comparison_form"):
            selected_paper_ids = st.multiselect(
                "Select 2–5 papers",
                options=list(available_papers.keys()),
                format_func=lambda paper_id: available_papers[paper_id],
                max_selections=5,
                help="Each paper is retrieved independently.",
            )

            comparison_question = st.text_area(
                "What should be compared?",
                placeholder=(
                    "Example: Compare the machine learning models, "
                    "datasets, evaluation metrics, and reported results."
                ),
                height=110,
            )

            comparison_top_k = st.slider(
                "Evidence passages per paper",
                min_value=2,
                max_value=5,
                value=3,
            )

            compare_submitted = st.form_submit_button(
                "Compare papers",
                type="primary",
                use_container_width=True,
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

                    st.divider()
                    st.subheader("Research Comparison")
                    st.markdown(comparison_answer)
                    st.download_button(
                        label="Download Research Brief (.md)",
                        data=research_brief,
                        file_name="research_brief.md",
                        mime="text/markdown",
                        type="primary",
                        use_container_width=True,
                    )

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

                    st.subheader("Source Citations")
                    render_citations(citation_metadata)

                    st.subheader("Retrieved Evidence")
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


with library_tab:
    st.subheader("Manage your research papers")

    st.write(
        "Upload PDF papers to add them to your local corpus. "
        "New papers are indexed alongside existing papers."
    )

    uploaded_files = st.file_uploader(
        "Upload research papers",
        type=["pdf"],
        accept_multiple_files=True,
        help="PDF files only. Existing filenames are skipped.",
    )

    if st.button(
        "Upload and index papers",
        type="primary",
        use_container_width=True,
        disabled=not uploaded_files,
    ):
        try:
            saved_paths, skipped_files = save_uploaded_pdfs(
                uploaded_files or []
            )

            if skipped_files:
                st.warning(
                    "Skipped: " + ", ".join(skipped_files)
                )

            if saved_paths:
                try:
                    with st.spinner(
                        "Rebuilding the corpus index. This may take a while..."
                    ):
                        get_retrieval_stack.clear()
                        index_corpus()
                        get_retrieval_stack.clear()
                except Exception:
                    # Rebuild is not transactional: remove only this upload
                    # batch and restore the prior index from remaining PDFs.
                    for path in saved_paths:
                        path.unlink(missing_ok=True)
                    try:
                        get_retrieval_stack.clear()
                        if list(PAPERS_DIR.glob("*.pdf")):
                            index_corpus()
                    finally:
                        get_retrieval_stack.clear()
                    raise

                st.success(f"Successfully indexed {len(saved_paths)} new PDF(s).")
                st.rerun()

            elif not skipped_files:
                st.info("Choose at least one PDF to upload.")

        except Exception as exc:
            st.error(
                "Indexing failed. Check the terminal for details and "
                "confirm that your ingestion and vector-store modules work."
            )
            st.exception(exc)

    st.divider()
    st.subheader("Current library")

    pdf_files = sorted(PAPERS_DIR.glob("*.pdf"))

    if pdf_files:
        for pdf_file in pdf_files:
            st.write(f"📄 **{pdf_file.name}**")
    else:
        st.info("No PDF papers have been added yet.")
