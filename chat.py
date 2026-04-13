from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


DB_FOLDER = "db"
COLLECTION_NAME = "company_knowledge"


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
        model="llava-v1.5-7b-llamafile", 
        base_url="http://localhost:1234/v1",
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


def main():
    vector_store = get_vector_store()
    retriever = vector_store.as_retriever(search_kwargs={"k": 2})
    llm = get_llm()

    chat_history = []

    print("Chatbot is ready. Type 'exit' to quit.\n")

    while True:
        question = input("You: ").strip()

        if question.lower() == "exit":
            break

        chat_history.append({"role": "user", "content": question})

        search_query = question

        docs = retriever.invoke(search_query)

        if not docs:
            answer = "I don't know based on the provided documents."
            print(f"\nBot: {answer}\n")
            chat_history.append({"role": "assistant", "content": answer})
            continue

        prompt = build_prompt(question, docs, chat_history)
        response = llm.invoke(prompt)
        answer = response.content.strip()

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