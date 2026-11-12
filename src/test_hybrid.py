from hybrid import reciprocal_rank_fusion


dense_results = [
    {"chunk_id": 10, "score": 0.90},
    {"chunk_id": 20, "score": 0.85},
    {"chunk_id": 30, "score": 0.80},
]

bm25_results = [
    {"chunk_id": 30, "score": 9.50},
    {"chunk_id": 10, "score": 8.80},
    {"chunk_id": 40, "score": 7.20},
]

fused_results = reciprocal_rank_fusion(
    [dense_results, bm25_results]
)

print("\n========== RRF TEST ==========")

for rank, result in enumerate(fused_results, start=1):
    print(
        f"Rank {rank}: "
        f"chunk={result['chunk_id']} "
        f"RRF={result['rrf_score']:.4f}"
    )

def test_rrf_handles_chunk_key_without_chunk_id_and_deduplicates():
    ranked = reciprocal_rank_fusion([
        [{"chunk_key": "p:stable", "paper_id": "p", "text": "same"}],
        [{"chunk_key": "p:stable", "paper_id": "p", "text": "same"}],
    ])
    assert len(ranked) == 1
    assert ranked[0]["chunk_key"] == "p:stable"


def test_rrf_falls_back_to_text_identity_when_chunk_id_missing():
    ranked = reciprocal_rank_fusion([[{"paper_id": "p", "text": "passage"}]])
    assert ranked[0]["chunk_key"] == "p:passage"
