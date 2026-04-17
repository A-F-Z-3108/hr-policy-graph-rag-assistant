from pathlib import Path
import traceback

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

try:
    data_dir = Path("data")
    index_dir = "faiss_index"
    pdf_paths = list(data_dir.glob("*.pdf"))

    print("PDF files found:", len(pdf_paths), flush=True)

    all_docs = []

    for pdf_path in pdf_paths:
        print(f"Loading: {pdf_path.name}", flush=True)
        docs = PyPDFLoader(str(pdf_path)).load()

        for doc in docs:
            doc.metadata["source_file"] = pdf_path.name

        all_docs.extend(docs)

    print("Total pages loaded:", len(all_docs), flush=True)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150
    )

    chunks = splitter.split_documents(all_docs)
    print("Total chunks created:", len(chunks), flush=True)

    print("Creating embeddings...", flush=True)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    print("Building FAISS index...", flush=True)
    vectorstore = FAISS.from_documents(chunks, embeddings)

    print("Saving index...", flush=True)
    vectorstore.save_local(index_dir)

    print("DONE", flush=True)
    print("Index saved:", index_dir, flush=True)

except Exception as e:
    print("ERROR:", str(e), flush=True)
    traceback.print_exc()
