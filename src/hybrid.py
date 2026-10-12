
from typing import Any


def reciprocal_rank_fusion(
    ranked_lists: list[list[dict[str, Any]]],
    k: int = 60,
) -> list[dict[str, Any]]:
    fused_scores: dict[str, float] = {}
    documents: dict[str, dict[str, Any]] = {}

    for ranked_results in ranked_lists:
        for rank, result in enumerate(ranked_results, start=1):
            chunk_key = result.get(
                "chunk_key",
                f"{result.get('paper_id', 'unknown')}:{result['chunk_id']}",
            )

            fused_scores[chunk_key] = (
                fused_scores.get(chunk_key, 0.0)
                + 1.0 / (k + rank)
            )
            documents[chunk_key] = result

    ranked_keys = sorted(
        fused_scores,
        key=lambda key: fused_scores[key],
        reverse=True,
    )

    fused_results = []

    for chunk_key in ranked_keys:
        result = documents[chunk_key].copy()
        result["chunk_key"] = chunk_key
        result["rrf_score"] = fused_scores[chunk_key]
        fused_results.append(result)

    return fused_results
