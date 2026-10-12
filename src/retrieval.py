
from typing import Any

from qdrant_client import QdrantClient

from embeddings import load_model, embed_text
from vector_store import COLLECTION_NAME, DB_PATH


def create_client() -> QdrantClient:
    return QdrantClient(path=DB_PATH)


def retrieve(
    client: QdrantClient,
    model,
    query: str,
    top_k: int = 5,
) -> list[dict[str, Any]]:
    query_embedding = embed_text(model, query)

    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_embedding,
        limit=top_k,
        with_payload=True,
    ).points

    retrieved_chunks = []

    for result in results:
        payload = result.payload or {}

        paper_id = payload.get("paper_id", "unknown")
        local_chunk_id = payload.get("chunk_id", result.id)

        retrieved_chunks.append(
            {
                "point_id": result.id,
                "chunk_id": local_chunk_id,
                "chunk_key": f"{paper_id}:{local_chunk_id}",
                "paper_id": paper_id,
                "paper_title": payload.get("paper_title", "Unknown paper"),
                "source_file": payload.get("source_file"),
                "score": result.score,
                "text": payload["text"],
                "section": payload.get("section"),
                "subsection": payload.get("subsection"),
                "page_start": payload.get("page_start"),
                "page_end": payload.get("page_end"),
                "token_count": payload.get("token_count"),
            }
        )

    return retrieved_chunks


if __name__ == "__main__":
    query = (
        "Which machine learning model performed "
        "best for predicting vehicle CO2 emissions?"
    )

    model = load_model()
    client = create_client()

    try:
        results = retrieve(client, model, query, top_k=5)

        for rank, result in enumerate(results, start=1):
            print(f"\n{'=' * 60}")
            print(f"Rank: {rank}")
            print(f"Paper: {result['paper_title']}")
            print(f"Paper ID: {result['paper_id']}")
            print(f"Chunk: {result['chunk_id']}")
            print(f"Section: {result['section']}")
            print(
                f"Pages: {result['page_start']}–"
                f"{result['page_end']}"
            )
            print(f"Score: {result['score']:.4f}")
            print(result["text"])
    finally:
        client.close()
