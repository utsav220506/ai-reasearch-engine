import re
from typing import Any

from rank_bm25 import BM25Okapi


def tokenize(text: str) -> list[str]:
    """
    Tokenize research text for BM25.

    Lowercasing and preserving technical alphanumeric
    terms gives BM25 useful lexical signals.
    """

    return re.findall(
        r"\b[a-zA-Z0-9]+(?:[-_][a-zA-Z0-9]+)*\b",
        text.lower(),
    )


def build_bm25_index(
    chunks: list[dict[str, Any]],
) -> BM25Okapi:
    """
    Build a BM25 index over chunk text.
    """

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
    """
    Retrieve chunks using BM25 lexical matching.
    """

    query_tokens = tokenize(query)

    scores = bm25.get_scores(query_tokens)

    ranked_indices = sorted(
        range(len(scores)),
        key=lambda i: scores[i],
        reverse=True,
    )[:top_k]

    results = []

    for index in ranked_indices:

        chunk = chunks[index]

        result = chunk.copy()

        result["score"] = float(
            scores[index]
        )

        results.append(result)

    return results


if __name__ == "__main__":

    from ingestion import (
        ingest_pdf,
        build_sections,
        build_chunks,
    )

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

    print(
        f"Built {len(chunks)} chunks."
    )

    print(
        "Building BM25 index..."
    )

    bm25 = build_bm25_index(chunks)

    query = (
        "Which machine learning model performed "
        "best for predicting vehicle CO2 emissions?"
    )

    results = bm25_retrieve(
        bm25,
        chunks,
        query,
        top_k=5,
    )

    print(
        f"\nQuery:\n{query}\n"
    )

    print(
        f"Retrieved {len(results)} chunks."
    )

    for rank, result in enumerate(
        results,
        start=1,
    ):

        print(
            f"\n{'=' * 70}"
        )

        print(
            f"RESULT {rank}"
        )

        print(
            f"BM25 score: "
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
            f"\n{result['text']}"
        )
        