# embed_pdf.py
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()                    # optional .env support

# ----------------------------------------------------------------------
# 1) CHOOSE YOUR EMBEDDING MODEL HERE
#    • If the env‑var EMBEDDING_MODEL is set it overrides DEFAULT_MODEL
#    • Good choices:  "BAAI/bge-base-en-v1.5"  or  "intfloat/e5-large-v2"
#    • Fast baseline: "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_MODEL = "abhinand/MedEmbed-large-v0.1"
# ----------------------------------------------------------------------

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders import PagedPDFSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS


def _embedding_function():
    model_name = os.getenv("EMBEDDING_MODEL", DEFAULT_MODEL)
    return HuggingFaceEmbeddings(model_name=model_name)


def embed_document(file_name: str,
                   file_folder: str = "pdf",
                   embedding_folder: str = "index") -> None:
    """
    Split one PDF into chunks, embed them, and save a FAISS index.
    """
    pdf_path = Path(file_folder) / file_name
    loader = PagedPDFSplitter(str(pdf_path))
    pages  = loader.load_and_split()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=450,
        chunk_overlap=100,
        length_function=len,
        is_separator_regex=False,
        separators=["\n\n", "\n", " ", ""],
    )
    docs = splitter.split_documents(pages)

    index = FAISS.from_documents(docs, _embedding_function())

    slug = file_name.replace(".pdf", "")        # <-- strip .pdf
    index.save_local(folder_path=embedding_folder,
                     index_name=f"{slug}.index")


def embed_all_pdf_docs(pdf_directory: str = "pdf",
                       embedding_folder: str = "index") -> None:
    """
    Embed **every** .pdf in pdf_directory into embedding_folder.
    """
    pdf_dir = Path(pdf_directory)
    if not pdf_dir.exists():
        raise FileNotFoundError(f"{pdf_dir} does not exist")

    pdf_files = [f for f in pdf_dir.iterdir() if f.suffix.lower() == ".pdf"]
    if not pdf_files:
        raise RuntimeError("No PDF files found to embed.")

    Path(embedding_folder).mkdir(parents=True, exist_ok=True)

    for pdf_file in pdf_files:
        print(f"Embedding {pdf_file.name} …")
        embed_document(pdf_file.name,
                       file_folder=pdf_directory,
                       embedding_folder=embedding_folder)
        print("Done!")


def get_all_index_files(index_directory: str = "index"):
    """
    Return a list of base names (without extension) for every FAISS index.
    """
    idx_dir = Path(index_directory)
    if not idx_dir.exists():
        raise FileNotFoundError(f"{idx_dir} does not exist")

    return sorted(
        f.stem.replace(".index", "")           # remove trailing .index
        for f in idx_dir.glob("*.index.faiss")
    )
