from embeddings import load_model
from retrieval import create_client, retrieve
from bm25 import build_bm25_index, bm25_retrieve
from hybrid import reciprocal_rank_fusion
from reranker import load_reranker, rerank
from ingestion import ingest_pdf, build_sections, build_chunks


PDF_PATH = "data/paper_1.pdf"


def build_retrievers():
    elements = ingest_pdf(PDF_PATH)

    enriched_elements = build_sections(elements)

    chunks = build_chunks(
        enriched_elements,
        max_chunk_tokens=400,
    )

    model = load_model()
    client = create_client()

    bm25 = build_bm25_index(chunks)

    reranker = load_reranker()

    return model, client, bm25, chunks, reranker


def hybrid_retrieve(
    query: str,
    model,
    client,
    bm25,
    chunks,
    reranker,
    top_k: int = 5,
):
    candidate_k = 10

    dense_results = retrieve(
        client,
        model,
        query,
        top_k=candidate_k,
    )

    bm25_results = bm25_retrieve(
        bm25,
        chunks,
        query,
        top_k=candidate_k,
    )

    fused_results = reciprocal_rank_fusion(
        [
            dense_results,
            bm25_results,
        ]
    )

    rerank_candidates = fused_results[:candidate_k]

    reranked_results = rerank(
        reranker,
        query,
        rerank_candidates,
        top_k=top_k,
    )

    return reranked_results


if __name__ == "__main__":
    model, client, bm25, chunks, reranker = build_retrievers()

    query = (
        "Which machine learning model performed "
        "best for predicting vehicle CO2 emissions?"
    )

    results = hybrid_retrieve(
        query,
        model,
        client,
        bm25,
        chunks,
        reranker,
        top_k=5,
    )

    print("\n========== HYBRID + RERANKER ==========")

    for rank, result in enumerate(
        results,
        start=1,
    ):
        print(
            f"\nRank {rank}"
            f"\nChunk: {result['chunk_id']}"
            f"\nRRF Score: {result['rrf_score']:.4f}"
            f"\nRerank Score: {result['rerank_score']:.4f}"
            f"\nSection: {result['section']}"
            f"\nPages: {result['page_start']}-{result['page_end']}"
            f"\nText: {result['text'][:300]}..."
        )

    client.close()