# RAG Evaluation Suite

Evaluation scripts for measuring the quality of the RAG chatbot's retrieval and answer generation.

## Prerequisites

- Python environment with project dependencies installed (`pip install -r requirements.txt`)
- Chroma vector store already built (`python ingest.py`)
- LM Studio running at `http://localhost:1234` (needed for answer eval and ablations, not retrieval-only)

## Files

| File | Purpose |
|------|---------|
| `questions.json` | Golden question set (48 questions: 29 direct, 8 paraphrased, 11 unanswerable) |
| `retrieval_eval.py` | Retrieval quality: Hit@1, Hit@3, Hit@5, MRR |
| `answer_eval.py` | End-to-end: RAG vs no-RAG baseline, generates labeling sheet |
| `score_labels.py` | Scores the hand-labeled CSV |
| `ablations.py` | Sweeps chunk_size, overlap, and top-k; optionally compares FAISS vs Chroma |

## Running the Evaluation

All commands should be run from the **project root** (`chat/`).

### Step 1: Retrieval Eval (no LLM needed)

```bash
python eval/retrieval_eval.py
```

Outputs: `eval/results/retrieval_k5.json` and a printed metrics table.

### Step 2: End-to-End Answer Eval (requires LM Studio)

```bash
python eval/answer_eval.py
```

Outputs:
- `eval/results/answers_rag.json` — RAG pipeline answers
- `eval/results/answers_baseline.json` — plain LLM answers (no retrieval)
- `eval/labeling_sheet.csv` — for manual labeling

### Step 3: Label and Score

1. Open `eval/labeling_sheet.csv` in a spreadsheet editor
2. Fill in the blank columns with `0` or `1`:
   - `correct_rag` — Is the RAG answer correct? (answerable questions)
   - `grounded_rag` — Is the RAG answer grounded in the retrieved context? (answerable questions)
   - `abstained_rag` — Did the RAG model correctly refuse to answer? (unanswerable questions)
   - `correct_baseline` — Is the baseline correct? For answerable: is the answer right? For unanswerable: did it correctly say "I don't know"?
3. Run:

```bash
python eval/score_labels.py
```

Outputs: `eval/results/scores.json` and a printed report.

### Step 4: Ablations (requires LM Studio for full run, retrieval-only without)

```bash
python eval/ablations.py
```

Sweeps:
- `chunk_size`: 256, 512, 1000
- `chunk_overlap`: 0, 50
- `top_k`: 3, 5

Outputs:
- `eval/results/ablations.json`
- `eval/results/ablations.md` (formatted table)

## Reproducibility

- LLM temperature is set to `0` for all eval runs
- Embedding model: `sentence-transformers/all-MiniLM-L6-v2` (deterministic)
- Questions and evidence strings are fixed in `questions.json`
- Results depend on the LM Studio model loaded at runtime
