from typing import Any

from qdrant_client import QdrantClient

from embeddings import load_model, embed_text
from vector_store import (
    COLLECTION_NAME,
    DB_PATH,
)


def create_client() -> QdrantClient:
    """
    Connect to the existing local Qdrant database.
    """

    return QdrantClient(
        path=DB_PATH
    )


def retrieve(
    client: QdrantClient,
    model,
    query: str,
    top_k: int = 5,
) -> list[dict[str, Any]]:
    """
    Retrieve the most semantically relevant chunks
    for a research question.
    """

    query_embedding = embed_text(
        model,
        query,
    )

    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_embedding,
        limit=top_k,
        with_payload=True,
    ).points

    retrieved_chunks = []

    for result in results:

        payload = result.payload

        retrieved_chunks.append(
            {
                "chunk_id": result.id,
                "score": result.score,
                "text": payload["text"],
                "section": payload["section"],
                "subsection": payload["subsection"],
                "page_start": payload["page_start"],
                "page_end": payload["page_end"],
                "token_count": payload["token_count"],
            }
        )

    return retrieved_chunks


if __name__ == "__main__":

    query = (
        "Which machine learning model performed "
        "best for predicting vehicle CO2 emissions?"
    )

    print("\nLoading embedding model...")

    model = load_model()

    print("\nConnecting to Qdrant...")

    client = create_client()

    print(
        f"\nQuery:\n{query}\n"
    )

    results = retrieve(
        client,
        model,
        query,
        top_k=5,
    )

    print(
        f"Retrieved {len(results)} chunks.\n"
    )

    for i, result in enumerate(
        results,
        start=1,
    ):

        print(
            f"\n{'=' * 70}"
        )

        print(
            f"RESULT {i}"
        )

        print(
            f"Similarity score: "
            f"{result['score']:.4f}"
        )

        print(
            f"Section: "
            f"{result['section']}"
        )

        print(
            f"Subsection: "
            f"{result['subsection']}"
        )

        print(
            f"Pages: "
            f"{result['page_start']} - "
            f"{result['page_end']}"
        )

        print(
            f"Tokens: "
            f"{result['token_count']}"
        )

        print(
            f"\n{result['text']}"
        )

    client.close()