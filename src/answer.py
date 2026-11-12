
import re

from hybrid_retrieval import build_retrievers, hybrid_retrieve
from generation import generate_answer


def answer_question(
    question: str,
    model,
    client,
    bm25,
    chunks,
    reranker,
    top_k: int = 5,
) -> dict:

    results = hybrid_retrieve(
        question,
        model,
        client,
        bm25,
        chunks,
        reranker,
        top_k=top_k,
    )

    if not results:
        return {
            "question": question,
            "answer": (
                "Insufficient evidence: no relevant research "
                "passages were retrieved."
            ),
            "citations": [],
            "evidence": [],
        }

    evidence = []

    for index, result in enumerate(results, start=1):
        evidence.append({
            "evidence_id": f"E{index}",
            "paper_id": result.get("paper_id", "unknown"),
            "paper_title": result.get("paper_title", "Unknown paper"),
            "source_file": result.get("source_file"),
            "chunk_id": result.get("chunk_id"),
            "chunk_key": result.get("chunk_key"),
            "section": result.get("section"),
            "page_start": result.get("page_start"),
            "page_end": result.get("page_end"),
            "text": result["text"],
        })

    answer = generate_answer(question, evidence)

    cited_ids = set(re.findall(r"\[E(\d+)\]", answer))

    citations = [
        {
            "evidence_id": item["evidence_id"],
            "paper_id": item["paper_id"],
            "paper_title": item["paper_title"],
            "source_file": item["source_file"],
            "chunk_id": item["chunk_id"],
            "section": item["section"],
            "page_start": item["page_start"],
            "page_end": item["page_end"],
        }
        for item in evidence
        if item["evidence_id"][1:] in cited_ids
    ]

    return {
        "question": question,
        "answer": answer,
        "citations": citations,
        "evidence": evidence,
    }


if __name__ == "__main__":
    model, client, bm25, chunks, reranker = build_retrievers()

    question = (
        "Which machine learning model achieved the highest "
        "classification accuracy for predicting engine performance, "
        "and what evidence supports the result?"
    )

    try:
        result = answer_question(
            question,
            model,
            client,
            bm25,
            chunks,
            reranker,
            top_k=5,
        )

        print("\n========== RESEARCH ANSWER ==========\n")
        print(result["answer"])

        print("\n========== RETRIEVED EVIDENCE ==========\n")

        for item in result["evidence"]:
            print(f"\n[{item['evidence_id']}] {item['paper_title']}")
            print(f"Paper ID: {item['paper_id']}")
            print(f"Chunk key: {item['chunk_key']}")
            print(f"Pages: {item['page_start']}-{item['page_end']}")
            print(f"Section: {item['section']}")
            print(f"\nPassage:\n{item['text']}")
            print("-" * 80)

        print("\n========== SOURCE REFERENCES ==========\n")

        for citation in result["citations"]:
            print(
                f"[{citation['evidence_id']}] "
                f"{citation['paper_title']} | "
                f"Paper ID: {citation['paper_id']} | "
                f"Pages: {citation['page_start']}-"
                f"{citation['page_end']} | "
                f"Section: {citation['section']}"
            )

    finally:
        client.close()
