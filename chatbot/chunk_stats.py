# Kyle Wright

import json
from typing import List
import statistics
import numpy as np

def load_chunks_from_json(json_path: str) -> List[dict]:
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)

def main():
    chunks = load_chunks_from_json("bupersinst_chunks.json")

    char_lengths = [len(chunk["text"]) for chunk in chunks]
    word_lengths = [len(chunk["text"].split()) for chunk in chunks]

    print(sorted(char_lengths, reverse=True))
    print(sorted(word_lengths, reverse=True))

    avg_char_len = sum(char_lengths) / len(char_lengths)
    min_char_len = min(char_lengths)
    max_char_len = max(char_lengths)
    med_char_len = statistics.median(char_lengths)
    char_len_90 = np.percentile(char_lengths, 90)

    avg_word_len = sum(word_lengths) / len(word_lengths)
    min_word_len = min(word_lengths)
    max_word_len = max(word_lengths)
    med_word_len = statistics.median(word_lengths)
    word_len_90 = np.percentile(word_lengths, 90)

    print(f"Chunks: {len(chunks)}")
    print(f"Average Length: {avg_char_len:.2f} characters")
    print(f"Median Length: {med_char_len} characters")
    print(f"Minimum Length: {min_char_len} characters")
    print(f"Maximum Length: {max_char_len} characters")
    print(f"90% Length: {char_len_90} characters")
    print(f"Average Length: {avg_word_len:.2f} words")
    print(f"Median Length: {med_word_len} words")
    print(f"Minimum Length: {min_word_len} words")
    print(f"Maximum Length: {max_word_len} words")
    print(f"90% Length: {word_len_90} words")


main()