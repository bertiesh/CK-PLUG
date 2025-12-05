"""
Prepare Natural Questions data for retrieval quality analysis using FAISS.
"""

import json
import argparse
from typing import List, Tuple, Optional

import numpy as np
import faiss
from datasets import load_dataset
from sentence_transformers import SentenceTransformer


def load_corpus_from_jsonl(corpus_path: str) -> Tuple[List[str], List[str]]:
    """
    Load a text corpus from a JSONL file.

    Each line is expected to be a JSON object with at least one of:
      - 'text'
      - 'contents'
      - 'passage'
      - 'body'

    Args:
        corpus_path: Path to the JSONL corpus.

    Returns:
        corpus_texts: List of passage texts.
        corpus_ids:   List of passage identifiers (string).
    """
    corpus_texts = []
    corpus_ids = []

    with open(corpus_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)

            text = (
                obj.get("text")
                or obj.get("contents")
                or obj.get("passage")
                or obj.get("body")
            )
            if not text:
                # Skip lines without any usable text field
                continue

            corpus_texts.append(text)
            # Use existing 'id' if present, else fall back to line index
            cid = obj.get("id", str(i))
            corpus_ids.append(str(cid))

    print(f"Loaded {len(corpus_texts)} passages from {corpus_path}")
    return corpus_texts, corpus_ids


def build_faiss_index(
    retriever: SentenceTransformer,
    corpus_texts: List[str],
    batch_size: int = 256,
) -> Tuple[faiss.Index, np.ndarray]:
    """
    Encode the corpus with SentenceTransformer and build a FAISS index (cosine similarity via L2-normalized IP).

    Args:
        retriever: SentenceTransformer model.
        corpus_texts: List of texts to encode and index.
        batch_size: Encoding batch size.

    Returns:
        index: FAISS index over the corpus embeddings.
        corpus_embeddings: (num_docs, dim) float32 numpy array.
    """
    print("Encoding corpus and building FAISS index...")
    all_embeddings = []

    for start in range(0, len(corpus_texts), batch_size):
        end = min(start + batch_size, len(corpus_texts))
        batch = corpus_texts[start:end]
        emb = retriever.encode(
            batch,
            batch_size=len(batch),
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        # Ensure float32 for FAISS
        emb = emb.astype("float32")
        all_embeddings.append(emb)

    corpus_embeddings = np.vstack(all_embeddings)
    # Normalize for cosine similarity
    faiss.normalize_L2(corpus_embeddings)

    dim = corpus_embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # inner product over normalized vectors = cosine
    index.add(corpus_embeddings)

    print(f"FAISS index built with {index.ntotal} vectors (dim={dim}).")
    return index, corpus_embeddings


from typing import Optional, Tuple, List

def extract_nq_qa(example) -> Optional[Tuple[str, List[str]]]:
    """
    Extract (question, answers) from a Natural Questions example.

    Supports both:
      - annotations as a dict: example["annotations"]["short_answers"], ["yes_no_answer"]
      - annotations as a list of dicts: for ann in example["annotations"]

    Returns:
        (question_text, answers_list) or None if no usable answer.
    """
    # Question field can be either a dict with 'text' or just a string
    q = example.get("question", "")
    if isinstance(q, dict):
        question = q.get("text", "")
    else:
        question = q

    if not question:
        return None

    answers: List[str] = []

    annotations = example.get("annotations", None)
    if annotations is None:
        return None

    # Case 1: annotations is a dict (your original code’s assumption)
    if isinstance(annotations, dict):
        short_answers = annotations.get("short_answers") or []
        # short_answers can be list of dicts (with 'text') or list of strings
        for sa in short_answers:
            if isinstance(sa, dict):
                text = sa.get("text")
            else:
                text = sa
            if text:
                answers.append(text)

        # Fallback to yes/no if no short answers
        if not answers:
            yna = annotations.get("yes_no_answer", -1)
            if yna != -1:
                answers.append("yes" if yna == 1 else "no")

    # Case 2: annotations is a list/tuple of dicts
    elif isinstance(annotations, (list, tuple)):
        for ann in annotations:
            if not isinstance(ann, dict):
                continue
            short_answers = ann.get("short_answers") or []
            for sa in short_answers:
                if isinstance(sa, dict):
                    text = sa.get("text")
                else:
                    text = sa
                if text:
                    answers.append(text)

        if not answers:
            # Try yes/no answers from the first annotation that has it
            for ann in annotations:
                if not isinstance(ann, dict):
                    continue
                yna = ann.get("yes_no_answer", -1)
                if yna != -1:
                    answers.append("yes" if yna == 1 else "no")
                    break

    # If we still have nothing, skip example
    if not answers:
        return None

    return question, answers


def prepare_nq_with_retrieval(
    output_path: str,
    corpus_path: str = None,
    max_examples: int = 1000,
    top_k: int = 10,
):
    """
    Prepare NQ dataset with retrieved contexts.

    Args:
        output_path: Where to save prepared data (JSONL).
        corpus_path: Path to Wikipedia corpus (JSONL with a 'text' field, or similar).
        max_examples: Number of NQ examples to prepare.
        top_k: Number of passages to retrieve per question.
    """
    # -------------------------------------------------------------------------
    # 1. Load NQ dataset
    # -------------------------------------------------------------------------
    print("Loading Natural Questions dataset (validation split)...")
    # If this fails, try: load_dataset("google-research-datasets/natural_questions", split="validation")
    nq = load_dataset("natural_questions", split="validation")

    # -------------------------------------------------------------------------
    # 2. Load retriever
    # -------------------------------------------------------------------------
    print("Loading retriever (BAAI/bge-base-en-v1.5)...")
    retriever = SentenceTransformer("BAAI/bge-base-en-v1.5")

    # -------------------------------------------------------------------------
    # 3. Load corpus and build FAISS index (if a corpus is provided)
    # -------------------------------------------------------------------------
    corpus_texts: List[str] = []
    corpus_ids: List[str] = []
    faiss_index: Optional[faiss.Index] = None

    if corpus_path:
        corpus_texts, corpus_ids = load_corpus_from_jsonl(corpus_path)
        if len(corpus_texts) == 0:
            print("Warning: corpus is empty; continuing without retrieval.")
        else:
            faiss_index, _ = build_faiss_index(retriever, corpus_texts)
    else:
        print("No corpus provided, proceeding without retrieval (empty contexts).")

    # -------------------------------------------------------------------------
    # 4. Iterate over NQ examples, retrieve contexts, and save
    # -------------------------------------------------------------------------
    prepared_data = []
    num_processed = 0

    for example in nq:
        if num_processed >= max_examples:
            break

        qa = extract_nq_qa(example)
        if qa is None:
            continue

        question, answers = qa

        # Retrieve contexts via FAISS if we have an index
        if faiss_index is not None and len(corpus_texts) > 0:
            q_emb = retriever.encode(
                [question], convert_to_numpy=True, show_progress_bar=False
            ).astype("float32")
            faiss.normalize_L2(q_emb)  # cosine similarity

            scores, indices = faiss_index.search(q_emb, top_k)
            indices = indices[0].tolist()
            scores = scores[0].tolist()

            retrieved_contexts = [corpus_texts[idx] for idx in indices]
            retrieved_doc_ids = [corpus_ids[idx] for idx in indices]
        else:
            retrieved_contexts = []
            retrieved_doc_ids = []
            scores = []

        item = {
            "question": question,
            # First answer as "answer" for convenience
            "answer": answers[0],
            # Also keep all short answers if you need them
            "all_answers": answers,
            # Retrieval outputs
            "retrieved_contexts": retrieved_contexts,
            "retrieved_doc_ids": retrieved_doc_ids,
            "retrieved_scores": scores,
        }

        prepared_data.append(item)
        num_processed += 1

        if num_processed % 100 == 0:
            print(f"Processed {num_processed} examples...")

    # -------------------------------------------------------------------------
    # 5. Save prepared data as JSONL
    # -------------------------------------------------------------------------
    print(f"Saving {len(prepared_data)} examples to {output_path} ...")
    with open(output_path, "w", encoding="utf-8") as f:
        for item in prepared_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Done. Saved {len(prepared_data)} examples to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--corpus_path", type=str, default=None)
    parser.add_argument("--max_examples", type=int, default=1000)
    parser.add_argument("--top_k", type=int, default=10)

    args = parser.parse_args()

    prepare_nq_with_retrieval(
        output_path=args.output_path,
        corpus_path=args.corpus_path,
        max_examples=args.max_examples,
        top_k=args.top_k,
    )
