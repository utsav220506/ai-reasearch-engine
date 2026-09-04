from pathlib import Path
from typing import Any, Dict, List, Optional

import pymupdf


# ==========================================================
# 1. LINE CLASSIFICATION
# ==========================================================

def classify_line(
    element: dict[str, Any],
    page_height: float,
) -> str:
    """
    Classify an extracted PDF line.

    Possible types:
        HEADER
        FOOTER
        CAPTION
        HEADING
        SUBHEADING
        BODY
    """

    text = element.get("text", "").strip()
    font = element.get("font", "")
    bbox = element.get("bbox", (0, 0, 0, 0))

    _, y0, _, y1 = bbox

    text_upper = text.upper()

    # ------------------------------------------------------
    # Footer
    # ------------------------------------------------------

    if y0 > page_height * 0.95:
        return "FOOTER"

    # ------------------------------------------------------
    # Header
    # ------------------------------------------------------

    if y1 < page_height * 0.05:
        return "HEADER"

    # ------------------------------------------------------
    # Figure / table captions
    # ------------------------------------------------------

    if (
        text_upper.startswith("FIGURE")
        or text_upper.startswith("TABLE")
    ):
        return "CAPTION"

    # ------------------------------------------------------
    # Explicit document labels
    # ------------------------------------------------------

    if text_upper in {
        "ABSTRACT",
        "INDEX TERMS",
        "KEYWORDS",
    }:
        return "HEADING"

    # ------------------------------------------------------
    # Numbered top-level sections
    #
    # Example:
    # I. INTRODUCTION
    # II. METHODS
    # III. RESULTS
    # ------------------------------------------------------

    if font == "FormataOTF-Bold":

        section_prefixes = (
            "I.",
            "II.",
            "III.",
            "IV.",
            "V.",
            "VI.",
            "VII.",
            "VIII.",
            "IX.",
            "X.",
        )

        if text_upper.startswith(section_prefixes):
            return "HEADING"

    # ------------------------------------------------------
    # Lettered subsections
    #
    # Example:
    # A. DATA PREPARATION
    # B. MACHINE LEARNING MODELS
    # ------------------------------------------------------

    if "MdIt" in font:
        return "SUBHEADING"

    # ------------------------------------------------------
    # Default
    # ------------------------------------------------------

    return "BODY"


# ==========================================================
# 2. PDF INGESTION
# ==========================================================

def ingest_pdf(
    pdf_path: str | Path,
) -> list[dict[str, Any]]:
    """
    Extract line-level information from a PDF.

    Each element contains:

        text
        page
        font
        size
        bbox
        type
    """

    pdf_path = Path(pdf_path)

    if not pdf_path.is_file():
        raise FileNotFoundError(
            f"PDF not found at {pdf_path}"
        )

    elements: list[dict[str, Any]] = []

    with pymupdf.open(pdf_path) as doc:

        for page_number, page in enumerate(
            doc,
            start=1,
        ):

            page_height = page.rect.height

            blocks = page.get_text("dict")["blocks"]

            for block in blocks:

                if "lines" not in block:
                    continue

                for line in block["lines"]:

                    text = ""

                    for span in line["spans"]:
                        text += span["text"]

                    text = text.strip()

                    if not text:
                        continue

                    first_span = line["spans"][0]

                    element = {
                        "text": text,
                        "page": page_number,
                        "font": first_span["font"],
                        "size": first_span["size"],
                        "bbox": line["bbox"],
                    }

                    element["type"] = classify_line(
                        element,
                        page_height,
                    )

                    elements.append(element)

    return elements


# ==========================================================
# 3. SECTION TRACKING
# ==========================================================

def build_sections(
    elements: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Add active section and subsection context
    to extracted elements.

    Example:

        HEADING
        III. MATERIALS AND METHODS

        SUBHEADING
        A. DATA PREPARATION

        BODY
        The dataset contains...

    becomes:

        section = III. MATERIALS AND METHODS
        subsection = A. DATA PREPARATION
    """

    enriched_elements: List[Dict[str, Any]] = []

    current_section: Optional[str] = None
    current_subsection: Optional[str] = None

    for element in elements:

        text = element.get("text", "").strip()
        elem_type = element.get("type")

        if not text:
            continue

        # --------------------------------------------------
        # New top-level section
        # --------------------------------------------------

        if elem_type == "HEADING":

            current_section = text
            current_subsection = None

        # --------------------------------------------------
        # New subsection
        # --------------------------------------------------

        elif elem_type == "SUBHEADING":

            if current_section is not None:
                current_subsection = text

        # --------------------------------------------------
        # Preserve original metadata
        # --------------------------------------------------

        enriched_element = element.copy()

        enriched_element["section"] = current_section
        enriched_element["subsection"] = current_subsection

        enriched_elements.append(
            enriched_element
        )

    return enriched_elements


# ==========================================================
# 4. TOKEN-AWARE CHUNKING
# ==========================================================

def build_chunks(
    elements: List[Dict[str, Any]],
    max_chunk_tokens: int = 400,
) -> List[Dict[str, Any]]:
    """
    Build structure-aware, token-limited chunks.

    Chunk boundaries occur when:

        1. Section changes.
        2. Subsection changes.
        3. Adding another element would exceed
           max_chunk_tokens.

    Headers and footers are excluded.

    Captions are preserved.

    The default limit is 400 tokens, leaving safety
    margin below BGE-base's 512-token limit.
    """

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        "BAAI/bge-base-en-v1.5"
    )

    chunks: List[Dict[str, Any]] = []

    current_text: List[str] = []

    current_section: Optional[str] = None
    current_subsection: Optional[str] = None

    current_page_start: Optional[int] = None
    current_page_end: Optional[int] = None

    def count_tokens(text: str) -> int:
        """
        Count tokens using the same tokenizer that
        will be used for our embedding model.
        """

        return len(
            tokenizer.encode(
                text,
                add_special_tokens=True,
            )
        )

    def flush_chunk() -> None:
        """
        Save the current chunk and reset state.
        """

        nonlocal current_text
        nonlocal current_page_start
        nonlocal current_page_end

        if not current_text:
            return

        chunk_text = "\n".join(
            current_text
        ).strip()

        chunks.append(
            {
                "text": chunk_text,
                "section": current_section,
                "subsection": current_subsection,
                "page_start": current_page_start,
                "page_end": current_page_end,
                "token_count": count_tokens(
                    chunk_text
                ),
            }
        )

        current_text = []

        current_page_start = None
        current_page_end = None

    # ------------------------------------------------------
    # Process elements
    # ------------------------------------------------------

    for element in elements:

        elem_type = element.get("type")

        text = element.get(
            "text",
            "",
        ).strip()

        if not text:
            continue

        # --------------------------------------------------
        # Remove document noise
        # --------------------------------------------------

        if elem_type in {
            "HEADER",
            "FOOTER",
        }:
            continue

        section = element.get(
            "section"
        )

        subsection = element.get(
            "subsection"
        )

        page = element.get(
            "page"
        )

        # --------------------------------------------------
        # Structural boundary
        # --------------------------------------------------

        structure_changed = (
            bool(current_text)
            and (
                section != current_section
                or subsection != current_subsection
            )
        )

        if structure_changed:
            flush_chunk()

        # --------------------------------------------------
        # Start new chunk
        # --------------------------------------------------

        if not current_text:

            current_section = section
            current_subsection = subsection

            current_page_start = page

        # --------------------------------------------------
        # Test proposed chunk size
        # --------------------------------------------------

        proposed_text = "\n".join(
            current_text + [text]
        )

        proposed_tokens = count_tokens(
            proposed_text
        )

        # --------------------------------------------------
        # If adding this element makes the chunk
        # too large, close the existing chunk.
        # --------------------------------------------------

        if (
            current_text
            and proposed_tokens > max_chunk_tokens
        ):

            flush_chunk()

            current_section = section
            current_subsection = subsection
            current_page_start = page

        # --------------------------------------------------
        # Add element to chunk
        # --------------------------------------------------

        current_text.append(text)

        if page is not None:
            current_page_end = page

    # ------------------------------------------------------
    # Save final chunk
    # ------------------------------------------------------

    flush_chunk()

    return chunks


# ==========================================================
# 5. TOKEN ANALYSIS
# ==========================================================

def analyze_chunk_tokens(
    chunks: List[Dict[str, Any]],
) -> None:
    """
    Analyze token distribution across generated chunks.
    """

    if not chunks:
        print("No chunks to analyze.")
        return

    token_lengths = [
        chunk["token_count"]
        for chunk in chunks
    ]

    over_limit = sum(
        tokens > 512
        for tokens in token_lengths
    )

    print("\n--- Token Analysis ---")

    print(
        f"Number of chunks: "
        f"{len(token_lengths)}"
    )

    print(
        f"Minimum tokens: "
        f"{min(token_lengths)}"
    )

    print(
        f"Maximum tokens: "
        f"{max(token_lengths)}"
    )

    print(
        f"Average tokens: "
        f"{sum(token_lengths) / len(token_lengths):.2f}"
    )

    print(
        f"Chunks over 512 tokens: "
        f"{over_limit}"
    )

    print(
        f"Percentage over limit: "
        f"{(over_limit / len(token_lengths)) * 100:.2f}%"
    )


# ==========================================================
# 6. MAIN
# ==========================================================

if __name__ == "__main__":

    pdf_path = "data/paper_1.pdf"

    # ------------------------------------------------------
    # Extract
    # ------------------------------------------------------

    elements = ingest_pdf(
        pdf_path
    )

    print(
        f"Extracted "
        f"{len(elements)} elements\n"
    )

    # ------------------------------------------------------
    # Add section context
    # ------------------------------------------------------

    enriched_elements = build_sections(
        elements
    )

    print(
        f"Built section context for "
        f"{len(enriched_elements)} elements\n"
    )

    # ------------------------------------------------------
    # Build token-aware chunks
    # ------------------------------------------------------

    chunks = build_chunks(
        enriched_elements,
        max_chunk_tokens=400,
    )

    # ------------------------------------------------------
    # Analyze token distribution
    # ------------------------------------------------------

    analyze_chunk_tokens(
        chunks
    )

    print(
        f"Built "
        f"{len(chunks)} chunks\n"
    )

    # ------------------------------------------------------
    # Inspect first 10 chunks
    # ------------------------------------------------------

    for i, chunk in enumerate(
        chunks[:10],
        start=1,
    ):

        print(
            f"\n===== CHUNK {i} ====="
        )

        print(
            f"Section: "
            f"{chunk['section']}"
        )

        print(
            f"Subsection: "
            f"{chunk['subsection']}"
        )

        print(
            f"Pages: "
            f"{chunk['page_start']} - "
            f"{chunk['page_end']}"
        )

        print(
            f"Tokens: "
            f"{chunk['token_count']}"
        )

        print(
            f"Characters: "
            f"{len(chunk['text'])}"
        )

        print("\nText:")

        print(
            chunk["text"]
        )