from sentence_transformers import CrossEncoder
import torch


MODEL_NAME = "BAAI/bge-reranker-base"


def get_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"

    return "cpu"


def load_reranker() -> CrossEncoder:
    device = get_device()

    print(f"Using reranker device: {device}")

    model = CrossEncoder(
        MODEL_NAME,
        device=device,
    )

    return model


def rerank(
    model: CrossEncoder,
    query: str,
    results: list[dict],
    top_k: int = 5,
) -> list[dict]:

    pairs = [
        [query, result["text"]]
        for result in results
    ]

    scores = model.predict(
        pairs,
        show_progress_bar=False,
    )

    reranked_results = []

    for result, score in zip(
        results,
        scores,
    ):
        reranked_result = result.copy()
        reranked_result["rerank_score"] = float(score)

        reranked_results.append(
            reranked_result
        )

    reranked_results.sort(
        key=lambda result: result["rerank_score"],
        reverse=True,
    )

    return reranked_results[:top_k]