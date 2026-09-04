from pathlib import Path
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams


COLLECTION_NAME = "research_papers"
VECTOR_SIZE = 768
DB_PATH = "data/qdrant"


def create_client() -> QdrantClient:
    """
    Create a local persistent Qdrant client.

    The database is stored on disk, so vectors survive
    after the Python process exits.
    """

    Path(DB_PATH).mkdir(
        parents=True,
        exist_ok=True,
    )

    return QdrantClient(
        path=DB_PATH
    )


def create_collection(
    client: QdrantClient,
) -> None:
    """
    Create the research paper vector collection.

    BGE-base produces 768-dimensional normalized vectors,
    so cosine similarity is used for retrieval.
    """

    existing_collections = [
        collection.name
        for collection in client.get_collections().collections
    ]

    if COLLECTION_NAME in existing_collections:
        print(
            f"Collection '{COLLECTION_NAME}' already exists."
        )
        return

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=VECTOR_SIZE,
            distance=Distance.COSINE,
        ),
    )

    print(
        f"Created collection: {COLLECTION_NAME}"
    )


def insert_chunks(
    client: QdrantClient,
    embedded_chunks: list[dict[str, Any]],
) -> None:
    """
    Insert embedded chunks and their metadata into Qdrant.
    """

    points = []

    for index, chunk in enumerate(
        embedded_chunks
    ):

        point = PointStruct(
            id=index,
            vector=chunk["embedding"],
            payload={
                "text": chunk["text"],
                "section": chunk["section"],
                "subsection": chunk["subsection"],
                "page_start": chunk["page_start"],
                "page_end": chunk["page_end"],
                "token_count": chunk["token_count"],
            },
        )

        points.append(point)

    client.upsert(
        collection_name=COLLECTION_NAME,
        points=points,
    )

    print(
        f"Inserted {len(points)} chunks."
    )


def get_collection_info(
    client: QdrantClient,
) -> None:

    info = client.get_collection(
        collection_name=COLLECTION_NAME
    )

    print("\n--- Collection Info ---")

    print(
        f"Points stored: "
        f"{info.points_count}"
    )

    print(
        f"Vector size: "
        f"{info.config.params.vectors.size}"
    )

    print(
        f"Distance: "
        f"{info.config.params.vectors.distance}"
    )

if __name__ == "__main__":

    from ingestion import (
        ingest_pdf,
        build_sections,
        build_chunks,
    )

    from embeddings import (
        load_model,
        embed_chunks,
    )

    pdf_path = "data/paper_1.pdf"

    # -------------------------
    # 1. Extract PDF
    # -------------------------

    print("Loading paper...")

    elements = ingest_pdf(pdf_path)

    enriched_elements = build_sections(
        elements
    )

    chunks = build_chunks(
        enriched_elements,
        max_chunk_tokens=400,
    )

    print(
        f"Built {len(chunks)} chunks."
    )

    # -------------------------
    # 2. Generate embeddings
    # -------------------------

    model = load_model()

    embedded_chunks = embed_chunks(
        model,
        chunks,
    )

    print(
        f"Generated {len(embedded_chunks)} embeddings."
    )

    # -------------------------
    # 3. Create Qdrant
    # -------------------------

    client = create_client()

    create_collection(client)

    # -------------------------
    # 4. Insert vectors
    # -------------------------

    insert_chunks(
        client,
        embedded_chunks,
    )

    # -------------------------
    # 5. Verify
    # -------------------------

    get_collection_info(client)