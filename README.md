# Company Policy Chatbot (RAG + Local LLM)

A lightweight chatbot that answers company policy questions using a **Retrieval-Augmented Generation (RAG)** pipeline with a **locally hosted LLM (LM Studio)**.

The system retrieves relevant information from internal documents and generates grounded responses, avoiding hallucinations and keeping everything **fully local (no external APIs)**.

---

## Overview

This project demonstrates how to build a practical RAG system end-to-end:

- Ingest company policy documents  
- Convert them into embeddings  
- Store them in a vector database  
- Retrieve relevant context for each query  
- Generate answers using a local language model  

The focus is on understanding how modern AI systems combine **search + generation**, instead of relying on LLMs alone.

---

## Architecture

```
User Query
   ↓
Embedding (MiniLM)
   ↓
Vector Search (Chroma)
   ↓
Relevant Document Chunks
   ↓
Local LLM (LM Studio)
   ↓
Final Answer
```

---

## Tech Stack

- Python  
- LangChain (RAG pipeline orchestration)  
- Chroma (vector database)  
- HuggingFace Embeddings (`all-MiniLM-L6-v2`)  
- LM Studio (local LLM server)  

### Model

- Local model via LM Studio (e.g. `DeepSeek-R1-Distill-Qwen-7B`)  
- Endpoint: `http://localhost:1234`  

---

## Key Features

- Fully local execution (no API keys, no external calls)  
- RAG-based responses grounded in company documents  
- Modular pipeline (easy to swap embeddings, models, or vector stores)  
- Simple CLI interface for querying  

---

## Running the Project

```bash
git clone https://github.com/Tuguzbay/company-policy-chatbot.git
cd company-policy-chatbot

python -m venv .venv
.venv\Scripts\activate   # Windows
# or
source .venv/bin/activate   # Mac/Linux

pip install -r requirements.txt
```

### Start Local Model

- Open LM Studio  
- Download and run a model  
- Make sure the server is running at:

```
http://localhost:1234
```

### Run chatbot

```bash
python chat.py
```

---

## Example

```
You: What is the vacation policy?
Bot: Employees are entitled to...
```

---

## Evaluation Results

The project includes a full evaluation suite (`eval/`) measuring retrieval quality, answer correctness, and hallucination resistance. See [`eval/README.md`](eval/README.md) for details.

| Metric | RAG | Baseline (no retrieval) |
|--------|-----|------------------------|
| Correctness | 0.81 (30/37) | 0.14 (5/37) |
| Groundedness | 0.89 (33/37) | N/A |
| Correct Abstention (unanswerable) | 1.00 (11/11) | 0.82 (9/11) |
| Hit@5 (retrieval) | 0.70 | N/A |
| MRR (retrieval) | 0.51 | N/A |

> 48 questions (37 answerable, 11 unanswerable), `deepseek-r1-distill-qwen-7b` at temperature 0, k=5, chunk size 500 / overlap 100. Answer labels (correctness, groundedness, abstention) were assigned by an LLM judge (Claude), not by hand.

---

## What I Learned

- How **RAG systems** work in practice (retrieval + generation)  
- How to use **LangChain** to structure LLM pipelines  
- How embeddings + vector search improve answer quality  
- How to run and integrate **local LLMs via LM Studio**  

---

## Future Improvements

- Web interface (Streamlit / React)  
- Better chunking and retrieval strategies  
- Conversation memory  
- Support for more document formats  

---

## Notes

This project focuses on building a practical AI system that combines retrieval and generation, with everything running locally for simplicity and control.