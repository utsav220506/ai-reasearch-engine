
import re
from pathlib import Path
from typing import Any

from rank_bm25 import BM25Okapi

from index_corpus import build_paper_chunks


def tokenize(text: str) -> list[str]:
    return re.findall(
        r"\b[a-zA-Z0-9]+(?:[-_][a-zA-Z0-9]+)*\b",
        text.lower(),
    )


def load_corpus_chunks() -> list[dict[str, Any]]:
    papers_dir = Path("data/papers")
    pdf_files = sorted(papers_dir.glob("*.pdf"))

    if not pdf_files:
        raise FileNotFoundError(
            f"No PDFs found in {papers_dir.resolve()}"
        )

    all_chunks = []

    for pdf_path in pdf_files:
        all_chunks.extend(build_paper_chunks(pdf_path))

    return all_chunks


def build_bm25_index(
    chunks: list[dict[str, Any]],
) -> BM25Okapi:
    corpus = [
        tokenize(chunk["text"])
        for chunk in chunks
    ]

    return BM25Okapi(corpus)


def bm25_retrieve(
    bm25: BM25Okapi,
    chunks: list[dict[str, Any]],
    query: str,
    top_k: int = 5,
) -> list[dict[str, Any]]:
    query_tokens = tokenize(query)
    scores = bm25.get_scores(query_tokens)

    ranked_indices = sorted(
        range(len(scores)),
        key=lambda i: scores[i],
        reverse=True,
    )[:top_k]

    results = []

    for index in ranked_indices:
        result = chunks[index].copy()

        result["chunk_key"] = (
            f"{result['paper_id']}:{result['chunk_id']}"
        )
        result["score"] = float(scores[index])

        results.append(result)

    return results


if __name__ == "__main__":
    print("Loading research corpus...")
    chunks = load_corpus_chunks()

    print(f"Total chunks: {len(chunks)}")
    print(
        f"Total papers: "
        f"{len(set(chunk['paper_id'] for chunk in chunks))}"
    )

    print("Building BM25 index...")
    bm25 = build_bm25_index(chunks)

    query = (
        "Which machine learning model performed best "
        "for predicting vehicle CO2 emissions?"
    )

    results = bm25_retrieve(
        bm25,
        chunks,
        query,
        top_k=5,
    )

    print(f"\nQuery: {query}")

    for rank, result in enumerate(results, start=1):
        print(f"\n{'=' * 70}")
        print(f"Rank: {rank}")
        print(f"Paper: {result['paper_title']}")
        print(f"Paper ID: {result['paper_id']}")
        print(f"Chunk: {result['chunk_id']}")
        print(f"BM25 score: {result['score']:.4f}")
        print(f"Section: {result.get('section')}")
        print(
            f"Pages: {result.get('page_start')}–"
            f"{result.get('page_end')}"
        )
        print(result["text"])
