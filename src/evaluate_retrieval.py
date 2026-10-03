import json

from embeddings import load_model
from retrieval import create_client, retrieve

from bm25 import build_bm25_index, bm25_retrieve
from hybrid import reciprocal_rank_fusion

from ingestion import ingest_pdf, build_sections, build_chunks


EVALUATION_FILE = "data/evaluation_questions.json"
PDF_PATH = "data/paper_1.pdf"


def load_questions():
    with open(EVALUATION_FILE, "r") as file:
        return json.load(file)


def find_relevant_rank(results, relevant_chunks):
    for rank, result in enumerate(results, start=1):
        if result["chunk_id"] in relevant_chunks:
            return rank
    return None


def recall_at_k(results, relevant_chunks, k):
    rank = find_relevant_rank(results, relevant_chunks)
    if rank is not None and rank <= k:
        return 1
    return 0


def reciprocal_rank(results, relevant_chunks):
    rank = find_relevant_rank(results, relevant_chunks)
    if rank is None:
        return 0.0
    return 1 / rank

def evaluate(questions, retrieve_function, name):
    recall_scores = []
    reciprocal_ranks = []

    print(f"\n========== {name} ==========")

    for index, item in enumerate(questions, start=1):

        results = retrieve_function(item["question"])

        relevant_chunks = item["relevant_chunks"]

        rank = find_relevant_rank(
            results,
            relevant_chunks,
        )

        recall = recall_at_k(
            results,
            relevant_chunks,
            k=5,
        )

        rr = reciprocal_rank(
            results,
            relevant_chunks,
        )

        recall_scores.append(recall)
        reciprocal_ranks.append(rr)

        rank_display = rank if rank is not None else "NOT FOUND"

        print(f"\nQ{index}: {item['question']}")
        print(f"Relevant chunks: {relevant_chunks}")
        print(f"Relevant rank: {rank_display}")
        print(f"Recall@5: {recall}")
        print(f"Reciprocal Rank: {rr:.4f}")

        print("Top results:")

        for result_rank, result in enumerate(results[:5], start=1):
            print(
                f"  {result_rank}. "
                f"chunk={result['chunk_id']} "
                f"section={result['section']}"
            )

    mean_recall = sum(recall_scores) / len(recall_scores)
    mean_reciprocal_rank = (
        sum(reciprocal_ranks) / len(reciprocal_ranks)
    )

    print(f"\n{name} Summary:")
    print(f"Recall@5: {mean_recall:.4f}")
    print(f"MRR:      {mean_reciprocal_rank:.4f}")

    return mean_recall, mean_reciprocal_rank


def main():

    questions = load_questions()

    # -------------------------
    # Build corpus
    # -------------------------

    elements = ingest_pdf(PDF_PATH)

    enriched_elements = build_sections(elements)

    chunks = build_chunks(
        enriched_elements,
        max_chunk_tokens=400,
    )

    # -------------------------
    # Initialize retrievers
    # -------------------------

    model = load_model()

    client = create_client()

    bm25 = build_bm25_index(chunks)

    # -------------------------
    # Dense retrieval
    # -------------------------

    def dense_search(question):

        return retrieve(
            client,
            model,
            question,
            top_k=5,
        )

    # -------------------------
    # BM25 retrieval
    # -------------------------

    def bm25_search(question):

        return bm25_retrieve(
            bm25,
            chunks,
            question,
            top_k=5,
        )

    # -------------------------
    # Hybrid retrieval
    # -------------------------

    def hybrid_search(question):

        candidate_k = 10

        dense_results = retrieve(
            client,
            model,
            question,
            top_k=candidate_k,
        )

        bm25_results = bm25_retrieve(
            bm25,
            chunks,
            question,
            top_k=candidate_k,
        )

        fused_results = reciprocal_rank_fusion(
            [
                dense_results,
                bm25_results,
            ]
        )

        return fused_results[:5]

    # -------------------------
    # Run evaluations
    # -------------------------

    dense_recall, dense_mrr = evaluate(
        questions,
        dense_search,
        "DENSE RETRIEVAL",
    )

    bm25_recall, bm25_mrr = evaluate(
        questions,
        bm25_search,
        "BM25 RETRIEVAL",
    )

    hybrid_recall, hybrid_mrr = evaluate(
        questions,
        hybrid_search,
        "HYBRID RRF",
    )

    # -------------------------
    # Final comparison
    # -------------------------

    print("\n========================================")
    print("FINAL RETRIEVAL COMPARISON")
    print("========================================")

    print(
        f"\n{'Retriever':<15}"
        f"{'Recall@5':<12}"
        f"{'MRR':<12}"
    )

    print("-" * 39)

    print(
        f"{'Dense':<15}"
        f"{dense_recall:<12.4f}"
        f"{dense_mrr:<12.4f}"
    )

    print(
        f"{'BM25':<15}"
        f"{bm25_recall:<12.4f}"
        f"{bm25_mrr:<12.4f}"
    )

    print(
        f"{'Hybrid RRF':<15}"
        f"{hybrid_recall:<12.4f}"
        f"{hybrid_mrr:<12.4f}"
    )

    client.close()


if __name__ == "__main__":
    main()