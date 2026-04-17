from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

data_dir = Path("data")
pdf_paths = list(data_dir.glob("*.pdf"))

all_docs = []

for pdf_path in pdf_paths:
    loader = PyPDFLoader(str(pdf_path))
    docs = loader.load()

    for doc in docs:
        doc.metadata["source_file"] = pdf_path.name

    all_docs.extend(docs)

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=150
)

chunks = splitter.split_documents(all_docs)

print("Total pages:", len(all_docs))
print("Total chunks:", len(chunks))

for i, chunk in enumerate(chunks[:3], 1):
    text = chunk.page_content[:400].encode("ascii", errors="ignore").decode()
    print(f"\n--- Chunk {i} ---")
    print("Source:", chunk.metadata.get("source_file"))
    print("Page:", chunk.metadata.get("page"))
    print(text)
