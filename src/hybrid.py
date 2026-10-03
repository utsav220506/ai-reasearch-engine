from typing import Any


def reciprocal_rank_fusion(
    ranked_lists: list[list[dict[str, Any]]],
    k: int = 60,
) -> list[dict[str, Any]]:
    """
    Combine multiple ranked result lists using Reciprocal Rank Fusion.

    RRF score:
        score(d) = sum(1 / (k + rank))

    rank starts at 1.
    """

    fused_scores: dict[int, float] = {}
    documents: dict[int, dict[str, Any]] = {}

    for ranked_results in ranked_lists:
        for rank, result in enumerate(ranked_results, start=1):
            chunk_id = result["chunk_id"]

            fused_scores[chunk_id] = (
                fused_scores.get(chunk_id, 0.0)
                + 1 / (k + rank)
            )

            documents[chunk_id] = result

    ranked_chunk_ids = sorted(
        fused_scores,
        key=lambda chunk_id: fused_scores[chunk_id],
        reverse=True,
    )

    fused_results = []

    for chunk_id in ranked_chunk_ids:
        result = documents[chunk_id].copy()
        result["rrf_score"] = fused_scores[chunk_id]
        fused_results.append(result)

    return fused_results