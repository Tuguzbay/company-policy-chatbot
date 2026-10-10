"""
End-to-end answer evaluation.

Runs every question through:
  1. Full RAG pipeline  -> eval/results/answers_rag.json
  2. Plain LLM (no retrieval) -> eval/results/answers_baseline.json

Then generates eval/labeling_sheet.csv for manual labeling.
"""

import json
import csv
import os
import re
import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chat import get_vector_store, get_llm, ask, TOP_K
from ingest import CHUNK_SIZE, CHUNK_OVERLAP


def load_questions(path="eval/questions.json"):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def run_rag(questions, retriever, llm):
    """Run all questions through the full RAG pipeline."""
    results = []
    for q in questions:
        qid = q["id"]
        question = q["question"]

        answer, docs = ask(question, retriever, llm)

        chunks = []
        for doc in docs:
            chunks.append({
                "chunk_id": doc.metadata.get("chunk_id"),
                "page": doc.metadata.get("page"),
                "document_name": doc.metadata.get("document_name"),
                "content": doc.page_content,
            })

        results.append({
            "id": qid,
            "question": question,
            "answerable": q["answerable"],
            "reference_answer": q["reference_answer"],
            "rag_answer": answer,
            "retrieved_chunks": chunks,
        })

        print(f"  [RAG] {qid}: {answer[:80]}...")

    return results


def run_baseline(questions, llm):
    """Run all questions through the LLM without any retrieval context."""
    results = []
    for q in questions:
        qid = q["id"]
        question = q["question"]

        # Simple prompt with no context
        prompt = f"""You are a company policy assistant.
Answer the following question based on your knowledge.
If you don't know the answer, say "I don't know."

Question: {question}"""

        response = llm.invoke(prompt)
        answer = re.sub(r"<think>.*?</think>", "", response.content, flags=re.DOTALL).strip()

        results.append({
            "id": qid,
            "question": question,
            "answerable": q["answerable"],
            "reference_answer": q["reference_answer"],
            "baseline_answer": answer,
        })

        print(f"  [BASE] {qid}: {answer[:80]}...")

    return results


def generate_labeling_sheet(questions, rag_results, baseline_results, path):
    """Generate CSV for manual labeling."""
    rag_lookup = {r["id"]: r["rag_answer"] for r in rag_results}
    base_lookup = {r["id"]: r["baseline_answer"] for r in baseline_results}

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "id", "question", "answerable", "reference_answer",
            "rag_answer", "baseline_answer",
            "correct_rag", "grounded_rag", "abstained_rag",
            "correct_baseline",
        ])

        for q in questions:
            writer.writerow([
                q["id"],
                q["question"],
                q["answerable"],
                q["reference_answer"],
                rag_lookup.get(q["id"], ""),
                base_lookup.get(q["id"], ""),
                "",  # correct_rag (manual)
                "",  # grounded_rag (manual)
                "",  # abstained_rag (manual)
                "",  # correct_baseline (manual)
            ])

    print(f"\nLabeling sheet saved to {path}")


def main():
    os.chdir(Path(__file__).resolve().parent.parent)

    print("Loading questions...")
    questions = load_questions()

    # Set temperature=0 for reproducibility
    llm = get_llm()
    llm.temperature = 0

    results_dir = Path("eval/results")
    results_dir.mkdir(parents=True, exist_ok=True)

    # --- RAG run ---
    rag_path = results_dir / "answers_rag.json"
    if rag_path.exists():
        # Delete the file to force a re-run of the RAG phase
        print(f"\nReusing existing RAG answers from {rag_path}")
        with open(rag_path, "r", encoding="utf-8") as f:
            rag_results = json.load(f)["results"]
    else:
        print("Initializing vector store...")
        vector_store = get_vector_store()
        retriever = vector_store.as_retriever(search_kwargs={"k": TOP_K})

        print(f"\n{'='*50}")
        print("Running RAG pipeline on all questions...")
        print(f"{'='*50}\n")
        rag_results = run_rag(questions, retriever, llm)

        rag_output = {
            "timestamp": datetime.now().isoformat(),
            "config": {"k": TOP_K, "chunk_size": CHUNK_SIZE, "chunk_overlap": CHUNK_OVERLAP, "temperature": 0},
            "results": rag_results,
        }
        with open(rag_path, "w", encoding="utf-8") as f:
            json.dump(rag_output, f, indent=2, ensure_ascii=False)
        print(f"\nRAG answers saved to {rag_path}")

    # --- Baseline run ---
    base_path = results_dir / "answers_baseline.json"
    if base_path.exists():
        # No retrieval involved, so only the LLM or the questions changing makes these stale
        print(f"\nReusing existing baseline answers from {base_path}")
        with open(base_path, "r", encoding="utf-8") as f:
            baseline_results = json.load(f)["results"]
    else:
        print(f"\n{'='*50}")
        print("Running baseline LLM (no retrieval) on all questions...")
        print(f"{'='*50}\n")
        baseline_results = run_baseline(questions, llm)

        base_output = {
            "timestamp": datetime.now().isoformat(),
            "config": {"temperature": 0, "retrieval": False},
            "results": baseline_results,
        }
        with open(base_path, "w", encoding="utf-8") as f:
            json.dump(base_output, f, indent=2, ensure_ascii=False)
        print(f"\nBaseline answers saved to {base_path}")

    # --- Labeling sheet ---
    generate_labeling_sheet(
        questions, rag_results, baseline_results,
        "eval/labeling_sheet.csv",
    )

    print("\nDone! Next steps:")
    print("  1. Open eval/labeling_sheet.csv")
    print("  2. Fill in correct_rag, grounded_rag, abstained_rag, correct_baseline (0/1)")
    print("  3. Run: python eval/score_labels.py")


if __name__ == "__main__":
    main()
