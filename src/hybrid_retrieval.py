
from embeddings import load_model
from retrieval import create_client, retrieve
from bm25 import build_bm25_index, bm25_retrieve, load_corpus_chunks
from hybrid import reciprocal_rank_fusion
from reranker import load_reranker, rerank


def build_retrievers(use_reranker=False):
    print("Loading research corpus...")
    chunks = load_corpus_chunks()

    if not chunks:
        raise RuntimeError("No research chunks found. Index PDFs first.")

    paper_ids = {chunk["paper_id"] for chunk in chunks}
    print(f"Total papers: {len(paper_ids)}")
    print(f"Total chunks: {len(chunks)}")

    model = load_model()
    client = create_client()
    bm25 = build_bm25_index(chunks)
    reranker = load_reranker() if use_reranker else None

    return model, client, bm25, chunks, reranker


def hybrid_retrieve(
    query,
    model,
    client,
    bm25,
    chunks,
    reranker=None,
    top_k=5,
):
    candidate_k = 10

    dense_results = retrieve(
        client, model, query, top_k=candidate_k
    )
    bm25_results = bm25_retrieve(
        bm25, chunks, query, top_k=candidate_k
    )

    fused_results = reciprocal_rank_fusion(
        [dense_results, bm25_results]
    )

    if reranker is None:
        return fused_results[:top_k]

    return rerank(
        reranker,
        query,
        fused_results[:candidate_k],
        top_k=top_k,
    )
