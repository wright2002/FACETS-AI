# Kyle Wright

import faiss
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
from typing import List, Tuple
import numpy as np
import json


LLM_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
EMBED_MODEL = "all-MiniLM-L6-v2"
# try to use GPU for faster processing
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TOP_K = 5
CHUNKS_JSON = "bupersinst_chunks.json"

llm_tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL)
llm_model = AutoModelForCausalLM.from_pretrained(LLM_MODEL).to(DEVICE)
embed_model = SentenceTransformer(EMBED_MODEL)


def load_chunks_from_json(json_path: str) -> List[dict]:
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)

def build_faiss_index(chunks: List[dict]) -> Tuple[faiss.IndexFlatL2, List[dict]]:
    texts = [chunk["text"] for chunk in chunks]
    vectors = embed_model.encode(texts)
    index = faiss.IndexFlatL2(vectors.shape[1])
    index.add(np.array(vectors))
    return index, vectors

def retrieve_relevant_chunks(query: str, chunks: List[dict], index, top_k: int) -> List[dict]:
    q_vec = embed_model.encode([query])
    D, I = index.search(np.array(q_vec), top_k)
    return [chunks[i] for i in I[0]]

def generate_answer(context: str, query: str) -> str:
    prompt = (f"### Human:\nAnswer the question concisely based on the BUPERSINST 1610.10G. CITE PAGE NUMBERS."
              f"\n{context}\n\n"
              f"{query}\n### Assistant:")
    inputs = llm_tokenizer(prompt, return_tensors="pt").to(DEVICE)
    input_token_count = inputs['input_ids'].shape[1]
    # print token counts for debugging
    print(f"[Token Count] Prompt: {input_token_count}")
    output = llm_model.generate(**inputs, max_new_tokens=150, do_sample=True, temperature=0.7)
    return llm_tokenizer.decode(output[0], skip_special_tokens=True).split("### Assistant:")[-1].strip()

def main():
    chunks = load_chunks_from_json(CHUNKS_JSON)
    index, _ = build_faiss_index(chunks)

    user_query = input("Ask a question: ")
    top_chunks = retrieve_relevant_chunks(user_query, chunks, index, TOP_K)

    # Combine content for prompt and reference
    context = "\n".join(f"[BUPERSINST 1610.10G Page {chunk['page']}]\n{chunk['text']}" for chunk in top_chunks)
    answer = generate_answer(context, user_query)

    print("\n--- Answer ---")
    print(answer)
#    print("Page numbers to review: ")
#    print(f"Page {chunk['page']}" for chunk in top_chunks)

# used for debug
    print("\n--- Context Used ---")
    print(context)

main()
