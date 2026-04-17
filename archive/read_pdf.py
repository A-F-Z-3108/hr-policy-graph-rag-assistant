from langchain_community.document_loaders import PyPDFLoader

file_path = "data/Employee’s Guide to the Family and Medical Leave Act.pdf"

loader = PyPDFLoader(file_path)
docs = loader.load()

print("Pages loaded:", len(docs))
print("Page number:", docs[0].metadata.get("page"))
print("Source:", docs[0].metadata.get("source"))
print(docs[0].page_content[:500])
