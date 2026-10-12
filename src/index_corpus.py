
from pathlib import Path
import re

from ingestion import ingest_pdf, build_sections, build_chunks
from embeddings import load_model, embed_chunks
from vector_store import create_client, create_collection, insert_chunks, get_collection_info


PAPERS_DIR = Path("data/papers")


def make_paper_id(pdf_path: Path) -> str:
    return re.sub(
        r"[^a-z0-9]+",
        "-",
        pdf_path.stem.lower(),
    ).strip("-")


def build_paper_chunks(pdf_path: Path) -> list[dict]:
    paper_id = make_paper_id(pdf_path)
    paper_title = pdf_path.stem.replace("_", " ").strip()

    elements = ingest_pdf(pdf_path)
    enriched_elements = build_sections(elements)
    chunks = build_chunks(enriched_elements, max_chunk_tokens=400)

    for chunk in chunks:
        chunk["paper_id"] = paper_id
        chunk["paper_title"] = paper_title
        chunk["source_file"] = pdf_path.name

    print(f"{paper_title}: {len(chunks)} chunks")
    return chunks


def main() -> None:
    pdf_files = sorted(PAPERS_DIR.glob("*.pdf"))

    if not pdf_files:
        raise FileNotFoundError(
            f"No PDFs found in {PAPERS_DIR.resolve()}"
        )

    paper_ids = [make_paper_id(path) for path in pdf_files]

    if len(paper_ids) != len(set(paper_ids)):
        raise ValueError(
            "Two PDF filenames produce the same paper ID. "
            "Rename the files so their stems are unique."
        )

    all_chunks = []

    for pdf_path in pdf_files:
        all_chunks.extend(build_paper_chunks(pdf_path))

    print(f"\nTotal papers: {len(pdf_files)}")
    print(f"Total chunks: {len(all_chunks)}")

    model = load_model()
    embedded_chunks = embed_chunks(model, all_chunks)

    client = create_client()

    try:
        create_collection(client)
        insert_chunks(client, embedded_chunks)
        get_collection_info(client)
    finally:
        client.close()


if __name__ == "__main__":
    main()
