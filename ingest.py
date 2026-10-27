# ============================================================
#  ingest.py — PDF extraction, chunking, and FAISS indexing
# ============================================================
# Run this once per new batch of PDFs.
# Usage:  python ingest.py                       (indexes all PDFs in data/pdfs/)
#         python ingest.py path/to/paper.pdf     (index a single file)


from pathlib import Path
import re
import fitz
import pickle
import os
import numpy as np
import pytesseract
from PIL import Image
import io

from config import DATA_PATH, CHUNK_SIZE, CHUNK_OVERLAP
from embeddings import load_embedding_model
import faiss

_NOISE_STRINGS = [
    "Scanned with CamScanner",
    "Downloaded from",
    "This content downloaded from",
]

# ── Admit-card / structured-form label sets ──────────────────────────────────
# These are column headers that fitz extracts separately from their values.
# When we detect this pattern we zip headers → values to produce labelled text.
_ADMIT_CARD_ROW_HEADERS = [
    "Date of Test",
    "Reporting Time",
    "Gate Closure Time",
    "Test Timing",
]

# Regex: matches the unlabelled data row fitz produces for admit-card tables
# e.g. "02-08-2004 TAG264170584 14/Apr/2026 07:30 AM 08:30 AM 9:00 AM - 12:10 PM"
_DATE_RE = re.compile(r'\b\d{2}[/-][A-Za-z]{3,9}[/-]\d{4}\b|\b\d{4}-\d{2}-\d{2}\b|\b\d{2}/\d{2}/\d{4}\b')
_TIME_RE = re.compile(r'\b\d{1,2}:\d{2}\s*(?:AM|PM)\b', re.IGNORECASE)


def clean_text(text: str) -> str:
    for noise in _NOISE_STRINGS:
        text = text.replace(noise, "")
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _label_structured_page(page_fitz) -> str:
    """
    Use fitz 'dict' mode to extract text blocks with their bounding-box
    positions, then reconstruct key→value pairs for form/table layouts.

    For admit cards the layout is:
      Left column  = labels  (Candidate Name, Date of Birth, …)
      Right column = values  (Samay Amrutlal Patel, 02-08-2004, …)

    We sort all text spans by (y, x) and heuristically pair them.
    """
    try:
        d = page_fitz.get_text("dict")
    except Exception:
        return page_fitz.get_text()

    # Collect all spans as (x0, y0, text)
    spans = []
    for block in d.get("blocks", []):
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []):
            line_text = " ".join(s["text"] for s in line.get("spans", [])).strip()
            if line_text:
                x0 = line["bbox"][0]
                y0 = line["bbox"][1]
                spans.append((round(y0), round(x0), line_text))

    spans.sort()   # sort by y (row) then x (column)

    # Group by approximate y-row (within 8px = same row)
    rows: list[list[str]] = []
    current_y = None
    current_row: list[str] = []
    for y, x, text in spans:
        if current_y is None or abs(y - current_y) <= 8:
            current_row.append(text)
            current_y = y
        else:
            if current_row:
                rows.append(current_row)
            current_row = [text]
            current_y = y
    if current_row:
        rows.append(current_row)

    # Reconstruct as labelled lines
    lines_out: list[str] = []
    i = 0
    while i < len(rows):
        row = rows[i]
        joined = " | ".join(row)

        # Detect unlabelled data rows (contain dates/times but no label words)
        has_date = bool(_DATE_RE.search(joined))
        has_time = bool(_TIME_RE.search(joined))
        is_data_row = (has_date or has_time) and not any(
            kw in joined for kw in ("Date of Test", "Reporting", "Gate Closure", "Test Timing", "Date of Birth")
        )

        if is_data_row:
            # Look back for a header row to pair with
            header_row = rows[i - 1] if i > 0 else []
            if header_row and any(kw in " ".join(header_row) for kw in _ADMIT_CARD_ROW_HEADERS):
                # Zip header tokens with value tokens
                tokens = joined.replace("|", " ").split()
                headers = header_row
                # Build explicit key: value pairs
                for h in headers:
                    h_clean = h.strip()
                    # Find the most likely matching token
                    if "Date of Test" in h_clean:
                        val = next((t for t in tokens if _DATE_RE.match(t)), "")
                        if val:
                            lines_out.append(f"Date of Test: {val}")
                    elif "Reporting" in h_clean:
                        times = _TIME_RE.findall(joined)
                        if times:
                            lines_out.append(f"Reporting Time: {times[0]}")
                    elif "Gate Closure" in h_clean:
                        times = _TIME_RE.findall(joined)
                        if len(times) >= 2:
                            lines_out.append(f"Gate Closure Time: {times[1]}")
                    elif "Test Timing" in h_clean:
                        # Test timing is a range like "9:00 AM - 12:10 PM"
                        timing_match = re.search(r'(\d{1,2}:\d{2}\s*(?:AM|PM)\s*-\s*\d{1,2}:\d{2}\s*(?:AM|PM))', joined, re.IGNORECASE)
                        if timing_match:
                            lines_out.append(f"Test Timing: {timing_match.group(1)}")
            else:
                lines_out.append(joined)
        else:
            lines_out.append(joined)

        i += 1

    return "\n".join(lines_out)


def _label_candidate_row(text: str) -> str:
    """
    fitz dumps the final data line of an admit card as one long string:
      'Samay Amrutlal Patel DT20267945693\niON Digital Zone...\n02-08-2004 TAG264170584 14/Apr/2026 07:30 AM 08:30 AM 9:00 AM - 12:10 PM'

    We detect and rewrite that last line with explicit labels so the LLM
    can answer "what is the date" / "what is the venue" correctly.
    """
    lines = text.split("\n")
    out = []
    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Detect the packed data line: has a date + multiple times
        dates = _DATE_RE.findall(line)
        times = _TIME_RE.findall(line)

        if dates and len(times) >= 2:
            # This is the key row — rewrite it explicitly
            # Pattern: DOB  RollNo  DateOfTest  ReportingTime  GateClosureTime  TestTimingStart - TestTimingEnd
            # Extract each field
            dob = ""
            date_of_test = ""
            for d in dates:
                # DOB is in DD-MM-YYYY or DD/MM/YYYY format; test date has a month name
                if re.search(r'[A-Za-z]', d):
                    date_of_test = d
                else:
                    dob = d

            # Roll / registration number: alphanumeric token that isn't a date/time
            tokens = line.split()
            reg_candidates = [
                t for t in tokens
                if re.match(r'^[A-Z]{2,}\d{5,}$', t) or re.match(r'^\d{10,}$', t)
            ]
            roll_no = reg_candidates[0] if reg_candidates else ""

            reporting_time = times[0] if len(times) > 0 else ""
            gate_closure   = times[1] if len(times) > 1 else ""

            timing_match = re.search(
                r'(\d{1,2}:\d{2}\s*(?:AM|PM)\s*-\s*\d{1,2}:\d{2}\s*(?:AM|PM))',
                line, re.IGNORECASE
            )
            test_timing = timing_match.group(1) if timing_match else ""

            labelled = []
            if dob:          labelled.append(f"Date of Birth: {dob}")
            if roll_no:      labelled.append(f"Roll Number: {roll_no}")
            if date_of_test: labelled.append(f"Date of Test: {date_of_test}")
            if reporting_time: labelled.append(f"Reporting Time: {reporting_time}")
            if gate_closure:   labelled.append(f"Gate Closure Time: {gate_closure}")
            if test_timing:    labelled.append(f"Test Timing: {test_timing}")

            out.append("\n".join(labelled))
        else:
            out.append(line)

    return "\n".join(out)


def extract_text_from_pdf(pdf_path) -> list[dict]:
    """
    Extract text page by page with structured labelling for form/table pages.
    Falls back to OCR for image-only pages.
    """
    if isinstance(pdf_path, str):
        pdf_path = Path(pdf_path)

    doc = fitz.open(str(pdf_path))
    pages = []

    for i, page in enumerate(doc):
        # Try structured extraction first
        text = _label_structured_page(page)

        # Also run the candidate-row labeller over the plain text
        plain = page.get_text()
        labelled_plain = _label_candidate_row(plain)

        # Merge both: structured layout + explicitly labelled data rows
        combined = text + "\n" + labelled_plain

        # Fallback to OCR if still empty
        if not combined.strip():
            pix = page.get_pixmap(dpi=200)
            img = Image.open(io.BytesIO(pix.tobytes()))
            combined = pytesseract.image_to_string(img, config="--psm 6")

        combined = clean_text(combined)

        if combined:
            pages.append({
                "source": pdf_path.name,
                "page": i + 1,
                "text": combined
            })

    return pages


def _split_into_sentences(text: str) -> list[str]:
    lines = text.split("\n")
    sentences = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if len(line) > 200:
            parts = re.split(r'(?<=[.!?])\s+', line)
            sentences.extend(p.strip() for p in parts if p.strip())
        else:
            sentences.append(line)
    return sentences


def chunk_pages(pages: list[dict], size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[dict]:
    """Chunk by accumulating lines up to `size` chars, with overlap."""
    if size <= 0:
        raise ValueError("Chunk size must be greater than zero.")
    overlap = max(0, min(overlap, size - 1))
    chunks = []

    for p in pages:
        sentences = _split_into_sentences(p["text"])
        current = ""

        for sent in sentences:
            # A single long line can exceed the limit by itself. Split it so
            # every indexed chunk respects the configured size.
            parts = [sent[i:i + size] for i in range(0, len(sent), size)] or [sent]
            for part in parts:
                if len(current) + len(part) + 1 <= size:
                    current = (current + "\n" + part).strip()
                else:
                    if current.strip():
                        chunks.append({"source": p["source"], "page": p["page"], "text": current.strip()})
                    overlap_text = current[-overlap:] if overlap and len(part) < size else ""
                    current = (overlap_text + "\n" + part).strip() if overlap_text else part

        if current.strip():
            chunks.append({"source": p["source"], "page": p["page"], "text": current.strip()})

    return chunks


def build_index(chunks: list[dict], index_dir: str | Path | None = None) -> None:
    chunks = [c for c in chunks if c["text"].strip()]

    if not chunks:
        raise ValueError("No valid text extracted from PDF(s).")

    texts = [c["text"] for c in chunks]
    model = load_embedding_model()
    embeddings = model.encode(texts, show_progress_bar=False)

    if len(embeddings.shape) == 1:
        embeddings = embeddings.reshape(1, -1)

    embeddings = embeddings.astype("float32")

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    target_dir = Path(index_dir) if index_dir else Path(DATA_PATH)
    target_dir.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(target_dir / "index.faiss"))

    with (target_dir / "meta.pkl").open("wb") as f:
        pickle.dump(chunks, f)

    print(f"[ingest] Indexed {len(chunks)} chunks -> {target_dir}")
