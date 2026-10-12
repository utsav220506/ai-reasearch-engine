
from embeddings import load_model
from retrieval import create_client, retrieve
from bm25 import (
    build_bm25_index,
    bm25_retrieve,
    load_corpus_chunks,
)
from hybrid import reciprocal_rank_fusion
from reranker import load_reranker, rerank


def build_retrievers():
    print("Loading research corpus...")
    chunks = load_corpus_chunks()

    paper_ids = {
        chunk["paper_id"]
        for chunk in chunks
    }

    print(f"Total papers: {len(paper_ids)}")
    print(f"Total chunks: {len(chunks)}")

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

    try:
        results = hybrid_retrieve(
            query,
            model,
            client,
            bm25,
            chunks,
            reranker,
            top_k=5,
        )

        print("\n========== MULTI-PAPER HYBRID + RERANKER ==========")

        for rank, result in enumerate(results, start=1):
            print(f"\n{'=' * 70}")
            print(f"Rank: {rank}")
            print(f"Paper: {result.get('paper_title', 'Unknown paper')}")
            print(f"Paper ID: {result.get('paper_id', 'unknown')}")
            print(f"Chunk: {result['chunk_id']}")
            print(f"Chunk key: {result.get('chunk_key', 'unknown')}")
            print(f"RRF score: {result['rrf_score']:.4f}")
            print(f"Rerank score: {result['rerank_score']:.4f}")
            print(f"Section: {result.get('section')}")
            print(
                f"Pages: {result.get('page_start')}–"
                f"{result.get('page_end')}"
            )
            print(f"\n{result['text'][:700]}")

    finally:
        client.close()