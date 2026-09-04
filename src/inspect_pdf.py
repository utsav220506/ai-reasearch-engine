from pathlib import Path

import pymupdf


def extract_lines(pdf_path: str | Path, page_number: int = 1):
    pdf_path = Path(pdf_path)

    if not pdf_path.is_file():
        raise FileNotFoundError(f"PDF not found at {pdf_path}")

    elements = []

    with pymupdf.open(pdf_path) as doc:
        page = doc[page_number - 1]

        blocks = page.get_text("dict")["blocks"]

        for block in blocks:
            if "lines" not in block:
                continue

            for line in block["lines"]:
                text = ""

                for span in line["spans"]:
                    text += span["text"]

                if not text.strip():
                    continue

                first_span = line["spans"][0]

                elements.append({
                    "text": text.strip(),
                    "page": page_number,
                    "font": first_span["font"],
                    "size": first_span["size"],
                    "bbox": line["bbox"],
                })

    return elements
from typing import Any, Dict


def classify_line(
    element: Dict[str, Any], page_height: float = 792.0
) -> str:

    bbox = element.get("bbox", (0, 0, 0, 0))

    _, y0, _, y1 = bbox

    # Bottom 5% → footer
    if y0 > page_height * 0.95:
        return "FOOTER"

    # Top 5% → header
    if y1 < page_height * 0.05:
        return "HEADER"

    # Everything else for now
    return "BODY"

if __name__ == "__main__":
    lines = extract_lines("data/paper_1.pdf", page_number=4)

    for element in lines[:20]:
     element["type"] = classify_line(element)
     print(element)