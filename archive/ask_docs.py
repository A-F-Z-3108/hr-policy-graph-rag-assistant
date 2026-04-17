from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

index_dir = "faiss_index"
question = "Can an employee take leave to care for a parent with a serious health condition?"

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

vectorstore = FAISS.load_local(
    index_dir,
    embeddings,
    allow_dangerous_deserialization=True
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
results = retriever.invoke(question)

print("Results found:", len(results))

for i, doc in enumerate(results, 1):
    text = doc.page_content[:500].encode("ascii", errors="ignore").decode()

    print(f"\n--- Result {i} ---")
    print("Source:", doc.metadata.get("source_file"))
    print("Page:", doc.metadata.get("page"))
    print(text)
