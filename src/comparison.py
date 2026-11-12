
from typing import Any

from hybrid import reciprocal_rank_fusion

# Patchable module-level seams keep the public module easy to unit test.
retrieve = build_bm25_index = bm25_retrieve = rerank = None

# Keep the comparison validation and evidence assembly importable in a
# lightweight environment; production dependencies are loaded at call time.
def _load_retrieval_dependencies(load_reranker=False):
    retrieve_impl = build_bm25_impl = bm25_retrieve_impl = rerank_impl = None
    if not globals()["retrieve"]:
        from retrieval import retrieve as retrieve_impl
    if not globals()["build_bm25_index"] or not globals()["bm25_retrieve"]:
        from bm25 import build_bm25_index as build_bm25_impl
        from bm25 import bm25_retrieve as bm25_retrieve_impl
    if not globals()["rerank"] and load_reranker:
        from reranker import rerank as rerank_impl
    return (
        globals()["retrieve"] or retrieve_impl,
        globals()["build_bm25_index"] or build_bm25_impl,
        globals()["bm25_retrieve"] or bm25_retrieve_impl,
        globals()["rerank"] or rerank_impl,
    )


def compare_papers(
    question: str,
    selected_paper_ids: list[str],
    model,
    client,
    chunks: list[dict[str, Any]],
    reranker=None,
    top_k_per_paper: int = 3,
) -> dict[str, Any]:

    if not isinstance(question, str) or not question.strip():
        raise ValueError("A non-empty comparison question is required.")

    if not isinstance(selected_paper_ids, list):
        raise ValueError("Paper selections must be provided as a list.")

    if not 2 <= len(selected_paper_ids) <= 5:
        raise ValueError("Select between two and five papers.")

    if any(
        not isinstance(paper_id, str) or not paper_id.strip()
        for paper_id in selected_paper_ids
    ):
        raise ValueError("Every selected paper must have a valid ID.")

    if len(set(selected_paper_ids)) != len(selected_paper_ids):
        raise ValueError("Duplicate paper selections are not allowed.")

    if (
        isinstance(top_k_per_paper, bool)
        or not isinstance(top_k_per_paper, int)
        or not 1 <= top_k_per_paper <= 10
    ):
        raise ValueError(
            "Evidence passages per paper must be between 1 and 10."
        )

    if not isinstance(chunks, list) or not chunks:
        raise RuntimeError("The indexed research corpus is empty.")

    indexed_paper_ids = {
        chunk.get("paper_id")
        for chunk in chunks
        if isinstance(chunk, dict)
    }

    missing_papers = [
        paper_id
        for paper_id in selected_paper_ids
        if paper_id not in indexed_paper_ids
    ]

    if missing_papers:
        raise RuntimeError(
            "The following selected papers are missing from the "
            "indexed corpus: " + ", ".join(missing_papers)
            + ". Reindex the research corpus before comparing."
        )

    retrieve, build_bm25_index, bm25_retrieve, rerank = _load_retrieval_dependencies(
        load_reranker=reranker is not None
    )
    all_evidence = []
    paper_results = []

    for paper_id in selected_paper_ids:
        paper_chunks = [
            chunk for chunk in chunks
            if chunk.get("paper_id") == paper_id
        ]

        if not paper_chunks:
            raise RuntimeError(
                f"No indexed chunks were found for paper {paper_id}."
            )

        paper_bm25 = build_bm25_index(paper_chunks)

        candidate_k = top_k_per_paper * 3

        dense_results = retrieve(
            client,
            model,
            question,
            top_k=candidate_k,
            paper_id=paper_id,
        )

        lexical_results = bm25_retrieve(
            paper_bm25,
            paper_chunks,
            question,
            top_k=candidate_k,
        )

        fused_results = reciprocal_rank_fusion(
            [dense_results, lexical_results]
        )

        if reranker is not None and fused_results:
            selected_results = rerank(
                reranker,
                question,
                fused_results,
                top_k=top_k_per_paper,
            )
        else:
            selected_results = fused_results[:top_k_per_paper]

        selected_results = [
            result for result in selected_results
            if isinstance(result, dict)
            and isinstance(result.get("text"), str)
            and result["text"].strip()
        ]

        if not selected_results:
            raise RuntimeError(
                f"No relevant evidence was retrieved for "
                f"{paper_chunks[0].get('paper_title', paper_id)} "
                f"({paper_id}). Try another question or reindex the paper."
            )

        paper_title = paper_chunks[0].get(
            "paper_title", paper_id
        )

        paper_results.append({
            "paper_id": paper_id,
            "paper_title": paper_title,
            "retrieved_count": len(selected_results),
        })

        all_evidence.extend(selected_results)

    evidence = []

    for index, result in enumerate(all_evidence, start=1):
        evidence.append({
            "evidence_id": f"E{index}",
            "paper_id": result.get("paper_id", "unknown"),
            "paper_title": result.get(
                "paper_title", "Unknown paper"
            ),
            "source_file": result.get("source_file"),
            "chunk_id": result.get("chunk_id"),
            "chunk_key": result.get("chunk_key"),
            "section": result.get("section"),
            "page_start": result.get("page_start"),
            "page_end": result.get("page_end"),
            "text": result["text"],
        })

    represented_papers = {
        item["paper_id"] for item in evidence
    }

    if represented_papers != set(selected_paper_ids):
        raise RuntimeError(
            "The retrieved evidence does not cover every selected paper. "
            "The comparison was stopped to avoid an incomplete result."
        )

    return {
        "question": question.strip(),
        "papers": paper_results,
        "evidence": evidence,
    }
