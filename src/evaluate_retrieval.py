import json
from pathlib import Path

from embeddings import load_model
from retrieval import create_client, retrieve


EVALUATION_FILE = (
    "data/evaluation_questions.json"
)


def load_questions() -> list[dict]:
    path = Path(EVALUATION_FILE)

    with open(path, "r") as file:
        return json.load(file)


def recall_at_k(
    results: list[dict],
    relevant_sections: list[str],
    k: int,
) -> int:

    top_results = results[:k]

    for result in top_results:

        if result["section"] in relevant_sections:
            return 1

    return 0


def reciprocal_rank(
    results: list[dict],
    relevant_sections: list[str],
) -> float:

    for rank, result in enumerate(
        results,
        start=1,
    ):

        if result["section"] in relevant_sections:
            return 1 / rank

    return 0.0


def main():

    questions = load_questions()

    model = load_model()
    client = create_client()

    recall_scores = []
    reciprocal_ranks = []

    print("\n========== RETRIEVAL EVALUATION ==========\n")

    for index, item in enumerate(
        questions,
        start=1,
    ):

        question = item["question"]
        relevant_sections = item[
            "relevant_sections"
        ]

        results = retrieve(
            client,
            model,
            question,
            top_k=5,
        )

        recall = recall_at_k(
            results,
            relevant_sections,
            k=5,
        )

        rr = reciprocal_rank(
            results,
            relevant_sections,
        )

        recall_scores.append(recall)
        reciprocal_ranks.append(rr)

        print(f"Question {index}:")
        print(question)

        print(
            f"Expected sections: "
            f"{relevant_sections}"
        )

        print(
            f"Retrieved sections:"
        )

        for rank, result in enumerate(
            results,
            start=1,
        ):

            print(
                f"  {rank}. "
                f"{result['section']} "
                f"({result['score']:.4f})"
            )

        print(
            f"Recall@5: {recall}"
        )

        print(
            f"Reciprocal Rank: {rr:.4f}"
        )

        print("-" * 60)

    mean_recall = (
        sum(recall_scores)
        / len(recall_scores)
    )

    mean_reciprocal_rank = (
        sum(reciprocal_ranks)
        / len(reciprocal_ranks)
    )

    print("\n========== RESULTS ==========")

    print(
        f"Recall@5: "
        f"{mean_recall:.4f}"
    )

    print(
        f"MRR: "
        f"{mean_reciprocal_rank:.4f}"
    )

    client.close()


if __name__ == "__main__":
    main()