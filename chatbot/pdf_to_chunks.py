# Kyle Wright

import fitz
import re
import json
from pathlib import Path

def extract_pdf_with_pages(pdf_path):
    doc = fitz.open(pdf_path)
    pages = []
    for page_num in range(len(doc)):
        text = doc[page_num].get_text()
        pages.append((page_num + 1, text))
    return pages

import re

def split_paragraphs(text):
    # Split on blank lines or indent-style breaks
    paragraphs = re.split(r'\n{2,}|\n\s{4,}|\n \n', text)
    return [p.strip() for p in paragraphs if p.strip()]

def build_chunks(pages, max_chunk_chars=5000):
    chunks = []
    for page_number, page_text in pages:
        paragraphs = split_paragraphs(page_text)

        for paragraph in paragraphs:
            if len(paragraph) > max_chunk_chars:
                # Force-split long paragraphs into sub-chunks
                for i in range(0, len(paragraph), max_chunk_chars):
                    subchunk = paragraph[i:i + max_chunk_chars]
                    chunks.append({
                        "page": page_number,
                        "text": subchunk.strip()
                    })
            else:
                chunks.append({
                    "page": page_number,
                    "text": paragraph
                })

    return chunks


def save_chunks_to_json(chunks, output_path):
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)

def main():
    pdf_file_path = "BUPERSINST_1610-10.pdf"
    output_json = "bupersinst_chunks.json"

    if Path(pdf_file_path).exists():
        pages = extract_pdf_with_pages(pdf_file_path)
        chunks = build_chunks(pages)
        save_chunks_to_json(chunks, output_json)
        print(f"Extracted and saved {len(chunks)} chunks to '{output_json}'")
    else:
        print("PDF file not found. Please check the file path.")

main()