
from pathlib import Path
from typing import Any
from uuid import NAMESPACE_URL, uuid5

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams


COLLECTION_NAME = "research_papers_v2"
VECTOR_SIZE = 768
DB_PATH = "data/qdrant"


def create_client() -> QdrantClient:
    Path(DB_PATH).mkdir(parents=True, exist_ok=True)
    return QdrantClient(path=DB_PATH)


def create_collection(client: QdrantClient) -> None:
    existing = {
        collection.name
        for collection in client.get_collections().collections
    }

    if COLLECTION_NAME not in existing:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(
                size=VECTOR_SIZE,
                distance=Distance.COSINE,
            ),
        )
        print(f"Created collection: {COLLECTION_NAME}")
    else:
        print(f"Collection already exists: {COLLECTION_NAME}")


def insert_chunks(
    client: QdrantClient,
    embedded_chunks: list[dict[str, Any]],
) -> None:
    points = []

    for chunk in embedded_chunks:
        paper_id = chunk["paper_id"]
        chunk_id = chunk["chunk_id"]
        point_id = str(
            uuid5(NAMESPACE_URL, f"{paper_id}:{chunk_id}")
        )

        payload = {
            key: value
            for key, value in chunk.items()
            if key != "embedding"
        }

        points.append(
            PointStruct(
                id=point_id,
                vector=chunk["embedding"],
                payload=payload,
            )
        )

    if points:
        client.upsert(
            collection_name=COLLECTION_NAME,
            points=points,
        )

    print(f"Indexed {len(points)} chunks.")


def get_collection_info(client: QdrantClient) -> None:
    info = client.get_collection(collection_name=COLLECTION_NAME)

    print(f"Collection: {COLLECTION_NAME}")
    print(f"Points stored: {info.points_count}")
    print(f"Vector size: {info.config.params.vectors.size}")
    print(f"Distance: {info.config.params.vectors.distance}")
