"""
Ablation study: sweep chunk_size, chunk_overlap, and top-k,
then run the retrieval eval for each configuration.

Optionally compares FAISS vs Chroma (if langchain-community[faiss] is installed).

Writes a combined results table to eval/results/ablations.md
and raw data to eval/results/ablations.json.
"""

import json
import os
import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

from ingest import load_documents, DOCS_FOLDER
from eval.retrieval_eval import load_questions, evaluate_retrieval, compute_aggregate


CHUNK_SIZES = [256, 512, 1000]
CHUNK_OVERLAPS = [0, 50]
TOP_KS = [3, 5]


def build_chroma_store(chunks, collection_name, embeddings):
    """Build an in-memory Chroma vector store from chunks (no files to clean up on Windows)."""
    from langchain_chroma import Chroma

    store = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
    )
    store.add_documents(chunks)
    return store


def try_build_faiss_store(chunks, embeddings):
    """Try to build a FAISS store. Returns None if FAISS is not installed."""
    try:
        from langchain_community.vectorstores import FAISS

        store = FAISS.from_documents(chunks, embeddings)
        return store
    except ImportError:
        print("  [FAISS not installed — skipping FAISS comparison]")
        return None


def run_ablation():
    os.chdir(Path(__file__).resolve().parent.parent)

    print("Loading questions...")
    questions = load_questions()
    answerable_count = sum(1 for q in questions if q["answerable"])
    print(f"  {answerable_count} answerable questions\n")

    print("Loading raw documents...")
    documents = load_documents(DOCS_FOLDER)
    print(f"  {len(documents)} raw documents/pages\n")

    print("Initializing embeddings (this may take a moment)...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    all_results = []

    for chunk_size in CHUNK_SIZES:
        for chunk_overlap in CHUNK_OVERLAPS:
            # Skip invalid combos
            if chunk_overlap >= chunk_size:
                continue

            print(f"\n{'='*60}")
            print(f"chunk_size={chunk_size}  chunk_overlap={chunk_overlap}")
            print(f"{'='*60}")

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
            chunks = splitter.split_documents(documents)
            for i, chunk in enumerate(chunks):
                chunk.metadata["chunk_id"] = i

            print(f"  Created {len(chunks)} chunks")

            # Build Chroma store
            store = build_chroma_store(
                chunks, f"ablation_{chunk_size}_{chunk_overlap}", embeddings
            )

            for k in TOP_KS:
                retriever = store.as_retriever(search_kwargs={"k": k})

                print(f"\n  --- Chroma, k={k} ---")
                results = evaluate_retrieval(questions, retriever, k=k)
                agg = compute_aggregate(results)

                row = {
                    "vector_store": "Chroma",
                    "chunk_size": chunk_size,
                    "chunk_overlap": chunk_overlap,
                    "k": k,
                    "num_chunks": len(chunks),
                    **agg,
                }
                all_results.append(row)

            # Try FAISS comparison (only for default overlap=50 combos to save time)
            if chunk_overlap == 50:
                faiss_store = try_build_faiss_store(chunks, embeddings)
                if faiss_store:
                    for k in TOP_KS:
                        retriever = faiss_store.as_retriever(search_kwargs={"k": k})

                        print(f"\n  --- FAISS, k={k} ---")
                        results = evaluate_retrieval(questions, retriever, k=k)
                        agg = compute_aggregate(results)

                        row = {
                            "vector_store": "FAISS",
                            "chunk_size": chunk_size,
                            "chunk_overlap": chunk_overlap,
                            "k": k,
                            "num_chunks": len(chunks),
                            **agg,
                        }
                        all_results.append(row)

    # --- Write results ---
    results_dir = Path("eval/results")
    results_dir.mkdir(parents=True, exist_ok=True)

    # JSON
    output = {
        "timestamp": datetime.now().isoformat(),
        "ablations": all_results,
    }
    with open(results_dir / "ablations.json", "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    # Markdown table
    write_markdown_table(all_results, results_dir / "ablations.md")

    print(f"\n\nResults saved to:")
    print(f"  {results_dir / 'ablations.json'}")
    print(f"  {results_dir / 'ablations.md'}")


def write_markdown_table(rows, path):
    lines = [
        "# Ablation Results",
        "",
        f"Generated: {datetime.now().isoformat()}",
        "",
        "| Store | Chunk Size | Overlap | k | Chunks | Hit@1 | Hit@3 | Hit@5 | MRR |",
        "|-------|-----------|---------|---|--------|-------|-------|-------|-----|",
    ]

    for r in rows:
        lines.append(
            f"| {r['vector_store']} "
            f"| {r['chunk_size']} "
            f"| {r['chunk_overlap']} "
            f"| {r['k']} "
            f"| {r['num_chunks']} "
            f"| {r.get('hit_at_1', 0):.3f} "
            f"| {r.get('hit_at_3', 0):.3f} "
            f"| {r.get('hit_at_5', 0):.3f} "
            f"| {r.get('mrr', 0):.3f} |"
        )

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    run_ablation()
