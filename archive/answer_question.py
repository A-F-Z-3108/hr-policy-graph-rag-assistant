from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

index_dir = "faiss_index"

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = FAISS.load_local(
    index_dir,
    embeddings,
    allow_dangerous_deserialization=True
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

model = ChatOpenAI(model="gpt-5.4")
chat_history = []

def ask_clarifier(question, history_text):
    prompt = f"""
You are helping decide whether a user's latest HR-policy question is clear enough to answer.

Conversation history:
{history_text}

Latest user question:
{question}

If the question is clear enough, reply exactly:
CLEAR

If the question is too vague or missing key details, reply exactly in this format:
ASK: <one short clarifying question>

Do not answer the HR question itself.
"""
    response = model.invoke(prompt)
    return response.content.strip()

while True:
    question = input("\nEnter your question (or type exit): ").strip()

    if question.lower() == "exit":
        print("Goodbye.")
        break

    history_text = "\n".join(chat_history[-6:])

    clarifier = ask_clarifier(question, history_text)

    if clarifier.startswith("ASK:"):
        follow_up = clarifier.replace("ASK:", "", 1).strip()
        print("\n=== CLARIFYING QUESTION ===")
        print(follow_up)

        chat_history.append(f"User: {question}")
        chat_history.append(f"Assistant: {follow_up}")
        continue

    retrieval_query = question
    if history_text:
        retrieval_query = f"""
Conversation so far:
{history_text}

Latest question:
{question}
"""

    docs = retriever.invoke(retrieval_query)

    context_parts = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source_file")
        page = doc.metadata.get("page")
        context_parts.append(
            f"[Source {i}: {source}, page {page}]\n{doc.page_content}"
        )

    context = "\n\n".join(context_parts)

    prompt = f"""
You are an HR policy assistant.

Use the conversation history and the retrieved context to answer the user's latest question.
Answer only from the retrieved context.
If the context is not enough, say so.
At the end, list the source numbers you used.

Conversation history:
{history_text}

Retrieved context:
{context}

Latest question:
{question}
"""

    response = model.invoke(prompt)
    answer = response.content.encode("ascii", errors="ignore").decode()

    print("\n=== FINAL ANSWER ===")
    print(answer)

    chat_history.append(f"User: {question}")
    chat_history.append(f"Assistant: {answer}")
