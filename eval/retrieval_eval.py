"""
Retrieval evaluation: Hit@1, Hit@3, Hit@5, and MRR
for answerable questions in the golden question set.

A retrieved chunk counts as correct if it contains the
question's evidence string, comparing only lowercase letters and digits
(pypdf drops spaces between words, e.g. "beconducted").
"""

import json
import re
import os
import sys
from pathlib import Path
from datetime import datetime

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chat import get_vector_store, TOP_K
from ingest import CHUNK_SIZE, CHUNK_OVERLAP


def load_questions(path="eval/questions.json"):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def normalize(text):
    return re.sub(r"[^a-z0-9]", "", text.lower())


def evaluate_retrieval(questions, retriever, k=5):
    """Run retrieval for each answerable question and compute metrics."""
    answerable = [q for q in questions if q["answerable"]]
    results = []

    for q in answerable:
        qid = q["id"]
        question = q["question"]
        evidence = normalize(q["evidence"])

        docs = retriever.invoke(question)

        hits = []
        for i, doc in enumerate(docs):
            content = normalize(doc.page_content)
            hit = evidence in content
            hits.append({
                "rank": i + 1,
                "hit": hit,
                "chunk_id": doc.metadata.get("chunk_id"),
                "page": doc.metadata.get("page"),
                "preview": doc.page_content[:200],
            })

        # Compute per-question metrics
        hit_ranks = [h["rank"] for h in hits if h["hit"]]
        first_hit = min(hit_ranks) if hit_ranks else None

        result = {
            "id": qid,
            "question": question,
            "evidence": q["evidence"],
            "hit_at_1": 1 if first_hit == 1 else 0,
            "hit_at_3": 1 if first_hit is not None and first_hit <= 3 else 0,
            "hit_at_5": 1 if first_hit is not None and first_hit <= 5 else 0,
            "reciprocal_rank": (1.0 / first_hit) if first_hit else 0.0,
            "first_hit_rank": first_hit,
            "retrieved_chunks": hits,
        }
        results.append(result)

        status = f"Hit@1={'Y' if result['hit_at_1'] else 'N'}"
        print(f"  [{qid}] {status}  RR={result['reciprocal_rank']:.2f}  {question[:60]}")

    return results


def compute_aggregate(results):
    """Compute aggregate metrics across all questions."""
    n = len(results)
    if n == 0:
        return {}

    return {
        "num_questions": n,
        "hit_at_1": sum(r["hit_at_1"] for r in results) / n,
        "hit_at_3": sum(r["hit_at_3"] for r in results) / n,
        "hit_at_5": sum(r["hit_at_5"] for r in results) / n,
        "mrr": sum(r["reciprocal_rank"] for r in results) / n,
    }


def print_table(agg):
    """Print a formatted results table."""
    print("\n" + "=" * 50)
    print("RETRIEVAL EVALUATION RESULTS")
    print("=" * 50)
    print(f"  Questions evaluated:  {agg['num_questions']}")
    print(f"  Hit@1:                {agg['hit_at_1']:.3f}")
    print(f"  Hit@3:                {agg['hit_at_3']:.3f}")
    print(f"  Hit@5:                {agg['hit_at_5']:.3f}")
    print(f"  MRR:                  {agg['mrr']:.3f}")
    print("=" * 50)


def main():
    os.chdir(Path(__file__).resolve().parent.parent)

    print("Loading questions...")
    questions = load_questions()

    print("Initializing vector store...")
    vector_store = get_vector_store()
    retriever = vector_store.as_retriever(search_kwargs={"k": TOP_K})

    print(f"\nRunning retrieval eval (k={TOP_K})...\n")
    results = evaluate_retrieval(questions, retriever, k=TOP_K)

    agg = compute_aggregate(results)
    print_table(agg)

    # Save detailed results
    output = {
        "timestamp": datetime.now().isoformat(),
        "config": {"k": TOP_K, "chunk_size": CHUNK_SIZE, "chunk_overlap": CHUNK_OVERLAP},
        "aggregate": agg,
        "per_question": results,
    }

    out_path = Path("eval/results/retrieval_k5.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\nDetailed results saved to {out_path}")


if __name__ == "__main__":
    main()
