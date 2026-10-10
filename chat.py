from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import re


DB_FOLDER = "db"
COLLECTION_NAME = "company_knowledge"
TOP_K = 5


def get_vector_store():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=DB_FOLDER,
    )
    return vector_store


def get_llm():
    # needs to be replaced if another LLM is used
    return ChatOpenAI(
        model="deepseek-r1-distill-qwen-7b", 
        base_url="http://127.0.0.1:1234/v1",
        api_key="not-needed",
        temperature=0.1,
    )


def build_prompt(question: str, docs, chat_history) -> str:
    context_parts = []
    for doc in docs:
        source = doc.metadata.get("document_name", "unknown")
        page = doc.metadata.get("page", "unknown")
        context_parts.append(
            f"[Source: {source}, page: {page}]\n{doc.page_content}"
        )

    context = "\n\n".join(context_parts)

    history_text = ""
    for msg in chat_history[-6:]:
        history_text += f"{msg['role'].capitalize()}: {msg['content']}\n"

    prompt = f"""
You are a friendly and professional company policy assistant.

Use the conversation history to understand the user's intent.
Use ONLY the provided context to answer policy-related questions.

Rules:
- Do not make up details.
- Do not infer company policy beyond what is explicitly written.
- If the answer is only partially supported, say that clearly.
- If the answer is not in the context, say exactly: "I don't know based on the provided documents."
- Use clear, natural language that is easy to understand.
- Keep the tone warm, calm, and helpful.
- Do not sound robotic or overly formal.
- Avoid overly strong "yes" or "no" answers unless the context clearly supports them.
- Do not suggest actions unless explicitly supported by the context.
- End with a short Sources section listing document name and page only.

Conversation History:
{history_text}

Context:
{context}

User Question:
{question}
"""
    return prompt.strip()


def ask(question: str, retriever, llm, chat_history=None):
    """Retrieve context and answer one question. Returns (answer, docs)."""
    docs = retriever.invoke(question)
    if not docs:
        return "I don't know based on the provided documents.", docs

    prompt = build_prompt(question, docs, chat_history or [])
    answer = llm.invoke(prompt).content
    # reasoning models (e.g. deepseek-r1) prefix answers with <think>...</think>
    answer = re.sub(r"<think>.*?</think>", "", answer, flags=re.DOTALL).strip()
    return answer, docs


def main():
    vector_store = get_vector_store()
    retriever = vector_store.as_retriever(search_kwargs={"k": TOP_K})
    llm = get_llm()

    chat_history = []

    print("Chatbot is ready. Type 'exit' to quit.\n")

    while True:
        question = input("You: ").strip()

        if question.lower() == "exit":
            break

        chat_history.append({"role": "user", "content": question})

        answer, docs = ask(question, retriever, llm, chat_history)

        print(f"\nBot: {answer}\n")

        print("Retrieved sources:")
        for doc in docs:
            print(
                f"- {doc.metadata.get('document_name')} | page {doc.metadata.get('page')}"
            )
        print()

        chat_history.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    main()