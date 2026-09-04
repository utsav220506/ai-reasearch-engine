from sentence_transformers import SentenceTransformer
import torch

from ingestion import ingest_pdf, build_sections, build_chunks


MODEL_NAME = "BAAI/bge-base-en-v1.5"


def get_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"

    return "cpu"


def load_model() -> SentenceTransformer:
    device = get_device()

    print(f"Using device: {device}")

    model = SentenceTransformer(
        MODEL_NAME,
        device=device,
    )

    return model

def embed_text(
    model: SentenceTransformer,
    text: str,
) -> list[float]:

    embedding = model.encode(
        text,
        normalize_embeddings=True,
    )

    return embedding.tolist()
def embed_chunks(
    model: SentenceTransformer,
    chunks: list[dict],
) -> list[dict]:

    texts = [
        chunk["text"]
        for chunk in chunks
    ]

    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    embedded_chunks = []

    for chunk, embedding in zip(
        chunks,
        embeddings,
    ):
        embedded_chunk = chunk.copy()

        embedded_chunk["embedding"] = embedding.tolist()

        embedded_chunks.append(
            embedded_chunk
        )

    return embedded_chunks


if __name__ == "__main__":

    pdf_path = "data/paper_1.pdf"

    print("Loading paper...")

    elements = ingest_pdf(pdf_path)

    enriched_elements = build_sections(
        elements
    )

    chunks = build_chunks(
        enriched_elements,
        max_chunk_tokens=400,
    )

    print(f"Built {len(chunks)} chunks")

    model = load_model()

    embedded_chunks = embed_chunks(
        model,
        chunks,
    )

    print("\n--- Embedding Analysis ---")

    print(
        f"Number of embedded chunks: "
        f"{len(embedded_chunks)}"
    )

    print(
        f"Embedding dimensions: "
        f"{len(embedded_chunks[0]['embedding'])}"
    )

    print(
        f"Chunk 1 token count: "
        f"{embedded_chunks[0]['token_count']}"
    )

    print(
        f"Chunk 1 embedding sample:"
    )

    print(
        embedded_chunks[0]["embedding"][:10]
    )