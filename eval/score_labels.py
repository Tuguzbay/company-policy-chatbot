"""
Score the manually labeled CSV and report metrics.

Reads eval/labeling_sheet.csv (after you fill in the 0/1 columns)
and reports:
  - Correctness: RAG vs Baseline
  - Groundedness: RAG only
  - Correct-abstention rate on unanswerable questions: RAG vs Baseline
"""

import csv
import json
import sys
from pathlib import Path
from datetime import datetime


def load_labels(path="eval/labeling_sheet.csv"):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def safe_int(val):
    """Parse a label value, returning None if blank."""
    val = val.strip()
    if val == "":
        return None
    return int(val)


def score(rows):
    answerable = [r for r in rows if r["answerable"].lower() == "true"]
    unanswerable = [r for r in rows if r["answerable"].lower() == "false"]

    # --- Correctness (answerable questions only) ---
    rag_correct = [safe_int(r["correct_rag"]) for r in answerable]
    base_correct = [safe_int(r["correct_baseline"]) for r in answerable]

    rag_correct_valid = [v for v in rag_correct if v is not None]
    base_correct_valid = [v for v in base_correct if v is not None]

    rag_correctness = (
        sum(rag_correct_valid) / len(rag_correct_valid)
        if rag_correct_valid else None
    )
    base_correctness = (
        sum(base_correct_valid) / len(base_correct_valid)
        if base_correct_valid else None
    )

    # --- Groundedness (RAG answerable only) ---
    rag_grounded = [safe_int(r["grounded_rag"]) for r in answerable]
    rag_grounded_valid = [v for v in rag_grounded if v is not None]
    rag_groundedness = (
        sum(rag_grounded_valid) / len(rag_grounded_valid)
        if rag_grounded_valid else None
    )

    # --- Abstention on unanswerable (RAG and Baseline) ---
    rag_abstained = [safe_int(r["abstained_rag"]) for r in unanswerable]
    base_abstained_on_unans = []
    for r in unanswerable:
        # For baseline: correct_baseline=1 means it correctly said "I don't know"
        val = safe_int(r["correct_baseline"])
        if val is not None:
            base_abstained_on_unans.append(val)

    rag_abstained_valid = [v for v in rag_abstained if v is not None]
    rag_abstention = (
        sum(rag_abstained_valid) / len(rag_abstained_valid)
        if rag_abstained_valid else None
    )
    base_abstention = (
        sum(base_abstained_on_unans) / len(base_abstained_on_unans)
        if base_abstained_on_unans else None
    )

    return {
        "answerable_count": len(answerable),
        "unanswerable_count": len(unanswerable),
        "correctness_rag": rag_correctness,
        "correctness_baseline": base_correctness,
        "groundedness_rag": rag_groundedness,
        "abstention_rag": rag_abstention,
        "abstention_baseline": base_abstention,
        "labeled_answerable_rag": len(rag_correct_valid),
        "labeled_answerable_base": len(base_correct_valid),
        "labeled_unanswerable_rag": len(rag_abstained_valid),
        "labeled_unanswerable_base": len(base_abstained_on_unans),
    }


def print_report(s):
    print("\n" + "=" * 60)
    print("EVALUATION REPORT — Human-Labeled Scores")
    print("=" * 60)

    print(f"\n  Answerable questions:   {s['answerable_count']}")
    print(f"  Unanswerable questions: {s['unanswerable_count']}")

    print(f"\n  --- Correctness (answerable) ---")
    if s["correctness_rag"] is not None:
        print(f"  RAG:      {s['correctness_rag']:.3f}  ({s['labeled_answerable_rag']} labeled)")
    else:
        print(f"  RAG:      (not yet labeled)")
    if s["correctness_baseline"] is not None:
        print(f"  Baseline: {s['correctness_baseline']:.3f}  ({s['labeled_answerable_base']} labeled)")
    else:
        print(f"  Baseline: (not yet labeled)")

    print(f"\n  --- Groundedness (RAG, answerable) ---")
    if s["groundedness_rag"] is not None:
        print(f"  RAG:      {s['groundedness_rag']:.3f}")
    else:
        print(f"  RAG:      (not yet labeled)")

    print(f"\n  --- Correct Abstention (unanswerable) ---")
    if s["abstention_rag"] is not None:
        print(f"  RAG:      {s['abstention_rag']:.3f}  ({s['labeled_unanswerable_rag']} labeled)")
    else:
        print(f"  RAG:      (not yet labeled)")
    if s["abstention_baseline"] is not None:
        print(f"  Baseline: {s['abstention_baseline']:.3f}  ({s['labeled_unanswerable_base']} labeled)")
    else:
        print(f"  Baseline: (not yet labeled)")

    print("\n" + "=" * 60)


def main():
    import os
    os.chdir(Path(__file__).resolve().parent.parent)

    print("Loading labels from eval/labeling_sheet.csv...")
    rows = load_labels()

    scores = score(rows)
    print_report(scores)

    # Save machine-readable scores
    out = {
        "timestamp": datetime.now().isoformat(),
        "scores": scores,
    }
    out_path = Path("eval/results/scores.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\nScores saved to {out_path}")


if __name__ == "__main__":
    main()
