
import json
import time
from pathlib import Path
import statistics

from embeddings import load_model
from retrieval import create_client, retrieve
from bm25 import build_bm25_index, bm25_retrieve
from hybrid import reciprocal_rank_fusion
from reranker import load_reranker, rerank
from index_corpus import build_paper_chunks


EVALUATION_FILE = Path("data/evaluation_questions.json")
PAPERS_DIR = Path("data/papers")
REPORT_FILE = Path("reports/retrieval_benchmark.json")
K = 5
CANDIDATE_K = 10


def load_questions():
    with EVALUATION_FILE.open("r") as file:
        questions = json.load(file)

    if not questions:
        raise ValueError("Evaluation dataset is empty.")

    for index, item in enumerate(questions, start=1):
        if not item.get("question") or not item.get("relevant_chunks"):
            raise ValueError(
                f"Question {index} must contain a question and relevant_chunks."
            )

    return questions


def result_key(result):
    if result.get("chunk_key"):
        return str(result["chunk_key"])

    paper_id = result.get("paper_id")
    chunk_id = result.get("chunk_id")

    if paper_id is not None and chunk_id is not None:
        return f"{paper_id}:{chunk_id}"

    raise ValueError(f"Retrieved result lacks chunk identity: {result}")


def calculate_metrics(results, relevant_chunks, k=K):
    relevant = {str(chunk) for chunk in relevant_chunks}

    if not relevant:
        raise ValueError("Relevant chunk labels cannot be empty.")

    ranked_keys = [result_key(result) for result in results]
    top_keys = ranked_keys[:k]

    retrieved_relevant = relevant.intersection(top_keys)
    hit = int(bool(retrieved_relevant))
    recall = len(retrieved_relevant) / len(relevant)

    first_relevant_rank = next(
        (
            rank
            for rank, key in enumerate(ranked_keys, start=1)
            if key in relevant
        ),
        None,
    )

    reciprocal_rank = (
        1 / first_relevant_rank if first_relevant_rank else 0.0
    )

    dcg = sum(
        1 / __import__("math").log2(rank + 1)
        for rank, key in enumerate(top_keys, start=1)
        if key in relevant
    )

    ideal_relevant_count = min(len(relevant), k)
    idcg = sum(
        1 / __import__("math").log2(rank + 1)
        for rank in range(1, ideal_relevant_count + 1)
    )

    ndcg = dcg / idcg if idcg else 0.0

    return {
        "hit_at_5": hit,
        "recall_at_5": recall,
        "reciprocal_rank": reciprocal_rank,
        "ndcg_at_5": ndcg,
        "first_relevant_rank": first_relevant_rank,
    }


def evaluate(questions, retrieve_function, name):
    per_question = []
    latencies = []

    print(f"\n========== {name} ==========")

    # Warm up the retrieval path before measuring latency.
    retrieve_function(questions[0]["question"])

    for index, item in enumerate(questions, start=1):
        start = time.perf_counter()
        results = retrieve_function(item["question"])
        latency_ms = (time.perf_counter() - start) * 1000

        metrics = calculate_metrics(
            results,
            item["relevant_chunks"],
        )

        metrics["question"] = item["question"]
        metrics["latency_ms"] = round(latency_ms, 2)
        metrics["relevant_chunks"] = item["relevant_chunks"]

        per_question.append(metrics)
        latencies.append(latency_ms)

        rank = metrics["first_relevant_rank"]
        rank_display = rank if rank is not None else "NOT FOUND"

        print(f"\nQ{index}: {item['question']}")
        print(f"Relevant rank: {rank_display}")
        print(f"Hit@5:         {metrics['hit_at_5']}")
        print(f"Recall@5:      {metrics['recall_at_5']:.4f}")
        print(f"MRR:           {metrics['reciprocal_rank']:.4f}")
        print(f"nDCG@5:        {metrics['ndcg_at_5']:.4f}")
        print(f"Latency:       {latency_ms:.2f} ms")

    metric_names = [
        "hit_at_5",
        "recall_at_5",
        "reciprocal_rank",
        "ndcg_at_5",
    ]

    summary = {
        metric: sum(row[metric] for row in per_question) / len(per_question)
        for metric in metric_names
    }

    summary["average_latency_ms"] = sum(latencies) / len(latencies)
    summary["median_latency_ms"] = statistics.median(latencies)
    summary["query_count"] = len(questions)

    print(f"\n{name} SUMMARY")
    for metric, value in summary.items():
        print(f"{metric:<24}: {value:.4f}" if isinstance(value, float)
              else f"{metric:<24}: {value}")

    return {
        "summary": summary,
        "per_question": per_question,
    }


def main():
    questions = load_questions()

    pdf_files = sorted(PAPERS_DIR.glob("*.pdf"))

    if not pdf_files:
        raise FileNotFoundError(f"No PDFs found in {PAPERS_DIR.resolve()}")

    chunks = []

    for pdf_path in pdf_files:
        chunks.extend(build_paper_chunks(pdf_path))

    if not chunks:
        raise RuntimeError("No chunks were extracted from the PDFs.")

    print(f"Evaluation questions: {len(questions)}")
    print(f"Evaluation PDFs:      {len(pdf_files)}")
    print(f"Evaluation chunks:    {len(chunks)}")

    model = load_model()
    client = create_client()
    bm25 = build_bm25_index(chunks)
    reranker = load_reranker()

    def dense_search(question):
        return retrieve(client, model, question, top_k=K)

    def bm25_search(question):
        return bm25_retrieve(bm25, chunks, question, top_k=K)

    def hybrid_search(question):
        dense_results = retrieve(
            client, model, question, top_k=CANDIDATE_K
        )
        lexical_results = bm25_retrieve(
            bm25, chunks, question, top_k=CANDIDATE_K
        )

        fused = reciprocal_rank_fusion(
            [dense_results, lexical_results]
        )
        return fused[:K]

    def reranked_search(question):
        dense_results = retrieve(
            client, model, question, top_k=CANDIDATE_K
        )
        lexical_results = bm25_retrieve(
            bm25, chunks, question, top_k=CANDIDATE_K
        )

        fused = reciprocal_rank_fusion(
            [dense_results, lexical_results]
        )

        return rerank(
            reranker,
            question,
            fused[:CANDIDATE_K],
            top_k=K,
        )

    retrievers = {
        "dense": dense_search,
        "bm25": bm25_search,
        "hybrid_rrf": hybrid_search,
        "hybrid_rrf_reranker": reranked_search,
    }

    report = {
        "evaluation_questions": len(questions),
        "evaluation_pdfs": len(pdf_files),
        "evaluation_chunks": len(chunks),
        "k": K,
        "candidate_k": CANDIDATE_K,
        "results": {},
    }

    try:
        for name, function in retrievers.items():
            report["results"][name] = evaluate(
                questions,
                function,
                name.upper(),
            )

        print("\n========== FINAL COMPARISON ==========")
        print(
            f"{'Retriever':<25}"
            f"{'Recall@5':<12}"
            f"{'MRR':<12}"
            f"{'nDCG@5':<12}"
            f"{'Latency ms':<12}"
        )

        for name, result in report["results"].items():
            summary = result["summary"]
            print(
                f"{name:<25}"
                f"{summary['recall_at_5']:<12.4f}"
                f"{summary['reciprocal_rank']:<12.4f}"
                f"{summary['ndcg_at_5']:<12.4f}"
                f"{summary['average_latency_ms']:<12.2f}"
            )

        REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)
        with REPORT_FILE.open("w") as file:
            json.dump(report, file, indent=2)

        print(f"\nBenchmark report saved to: {REPORT_FILE}")

    finally:
        client.close()


if __name__ == "__main__":
    main()
