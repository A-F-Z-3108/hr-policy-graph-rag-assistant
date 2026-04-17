from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader

data_dir = Path("data")
pdf_paths = list(data_dir.glob("*.pdf"))

for pdf_path in pdf_paths:
    loader = PyPDFLoader(str(pdf_path))
    docs = loader.load()

    text = docs[0].page_content[:300].encode("ascii", errors="ignore").decode()

    print("\n---", pdf_path.name, "---")
    print("Pages:", len(docs))
    print("First page preview:")
    print(text)
