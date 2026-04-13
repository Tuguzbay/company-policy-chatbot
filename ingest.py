from pathlib import Path
from pypdf import PdfReader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


DOCS_FOLDER = Path("docs")
DB_FOLDER = "db"
COLLECTION_NAME = "company_knowledge"


def read_text_file(filepath: Path) -> str:
    """Read a normal text file and return one big string."""
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read()


def read_pdf_file(filepath: Path) -> list[Document]:
    """
    Read a PDF and return a list of Document objects.
    One Document per page so we can keep page metadata.
    """
    reader = PdfReader(str(filepath))
    pages = []

    for page_number, page in enumerate(reader.pages, start=1):
        page_text = page.extract_text()
        if page_text and page_text.strip():
            pages.append(
                Document(
                    page_content=page_text,
                    metadata={
                        "document_name": filepath.name,
                        "document_id": filepath.stem,
                        "page": page_number,
                        "source": str(filepath),
                    },
                )
            )

    return pages


def load_documents(folder: Path) -> list[Document]:
    """
    Load all supported files from the docs folder.
    Returns a list of LangChain Document objects.
    """
    documents = []

    for filepath in folder.iterdir():
        if filepath.suffix.lower() == ".txt":
            text = read_text_file(filepath)
            documents.append(
                Document(
                    page_content=text,
                    metadata={
                        "document_name": filepath.name,
                        "document_id": filepath.stem,
                        "page": None,
                        "source": str(filepath),
                    },
                )
            )

        elif filepath.suffix.lower() == ".pdf":
            documents.extend(read_pdf_file(filepath))

    return documents


def split_documents(documents: list[Document]) -> list[Document]:
    """
    Split large documents into smaller chunks.
    We keep metadata attached to every chunk.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
    )

    chunks = splitter.split_documents(documents)

    # Add chunk ids for easier debugging later
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i

    return chunks


def build_vector_store(chunks: list[Document]) -> None:
    """
    Create a persistent Chroma database from document chunks.
    """
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Fresh rebuild for now.
    # Later you can make this smarter.
    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=DB_FOLDER,
    )

    # Clear old data by deleting and recreating collection logic is more advanced.
    # For version 1, just add documents. If you re-run often, you may want to wipe db/ manually.
    vector_store.add_documents(chunks)

    print(f"Stored {len(chunks)} chunks in the vector database.")


def main():
    if not DOCS_FOLDER.exists():
        print("docs/ folder not found.")
        return

    documents = load_documents(DOCS_FOLDER)
    print(f"Loaded {len(documents)} raw documents/pages.")

    chunks = split_documents(documents)
    print(f"Created {len(chunks)} chunks.")

    if chunks:
        print("\nExample chunk:\n")
        print(chunks[0].page_content[:400])
        print("\nMetadata:")
        print(chunks[0].metadata)

    build_vector_store(chunks)


if __name__ == "__main__":
    main()