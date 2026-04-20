from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS


class HRState(TypedDict):
    question: str
    history: str
    needs_clarification: bool
    clarifying_question: str
    topic: str
    answer: str


# Load saved index
index_dir = "faiss_index"

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = FAISS.load_local(
    index_dir,
    embeddings,
    allow_dangerous_deserialization=True
)

# Main LLM
model = ChatOpenAI(model="gpt-5.4")


def rule_based_clarity(question: str):
    # Quick rule-based clarity check
    q = question.lower().strip()

    if "user clarification:" in q:
        return True

    vague_patterns = [
        "what about leave",
        "what about that",
        "can they do that",
        "what if it changes",
        "what about the policy",
        "what about this",
        "hi",
        "hello"
    ]

    if any(p in q for p in vague_patterns):
        return False

    if "leave" in q and any(x in q for x in [
        "parent",
        "child",
        "spouse",
        "family",
        "serious health condition",
        "medical",
        "care for",
        "eligible",
        "employer"
    ]):
        return True

    clear_keywords = [
        "fmla", "absence",
        "accommodation", "reasonable accommodation", "disability", "ada",
        "osha", "inspection", "worker rights", "safety", "hazard"
    ]

    if any(k in q for k in clear_keywords):
        return True

    if len(q.split()) >= 8:
        return True

    return False


def check_clarity(state: HRState):
    # Decide if a clarifying question is needed
    question = state["question"]

    if "User clarification:" in question or rule_based_clarity(question):
        return {
            "needs_clarification": False,
            "clarifying_question": ""
        }

    prompt = f"""
The user's HR question is vague.

Question:
{question}

Ask one short clarifying question.
Reply only with the question itself.
"""
    response = model.invoke(prompt)

    return {
        "needs_clarification": True,
        "clarifying_question": response.content.strip()
    }


def ask_question(state: HRState):
    # Return the clarifying question
    return {"answer": f"CLARIFY: {state['clarifying_question']}"}


def classify_topic(state: HRState):
    # Route the question by topic
    question = state["question"]

    if "User clarification:" in question:
        question = question.split("User clarification:")[-1]

    q = question.lower()

    if any(x in q for x in [
        "osha",
        "inspection",
        "workplace safety",
        "hazard",
        "unsafe",
        "retaliation"
    ]):
        return {"topic": "osha"}

    if any(x in q for x in [
        "accommodation",
        "reasonable accommodation",
        "disability",
        "disabled",
        "ada",
        "medical condition"
    ]):
        return {"topic": "ada"}

    return {"topic": "leave"}


def source_matches(topic, source_name):
    # Match retrieved chunks to the selected topic
    s = (source_name or "").lower()

    if topic == "leave":
        return (
            "family and medical leave" in s or
            "leave act" in s or
            "fmla" in s
        )

    if topic == "ada":
        return (
            "accommodation" in s or
            "disabilities act" in s or
            "ada" in s
        )

    if topic == "osha":
        return (
            "osha" in s or
            "inspection" in s or
            "workers’ rights" in s or
            "workers' rights" in s
        )

    return False


def build_context(question, history, topic):
    # Retrieve and format supporting context
    retrieval_query = question
    if history:
        retrieval_query = f"""
Conversation so far:
{history}

Latest question:
{question}
"""

    docs = vectorstore.similarity_search(retrieval_query, k=12)

    filtered_docs = []
    for doc in docs:
        source = doc.metadata.get("source_file")
        if source_matches(topic, source):
            filtered_docs.append(doc)

    if not filtered_docs:
        filtered_docs = docs[:4]
    else:
        filtered_docs = filtered_docs[:4]

    context_parts = []
    for i, doc in enumerate(filtered_docs, 1):
        source = doc.metadata.get("source_file")
        page = doc.metadata.get("page")
        context_parts.append(
            f"[Source {i}: {source}, page {page}]\n{doc.page_content}"
        )

    return "\n\n".join(context_parts)


def answer_by_topic(state: HRState):
    # Generate the final answer from retrieved context
    context = build_context(state["question"], state["history"], state["topic"])

    prompt = f"""
You are an HR policy assistant.

Topic: {state["topic"]}

Answer only from the retrieved context.
If the context is not enough, say so.
At the end, list the source numbers you used.

Conversation history:
{state["history"]}

Retrieved context:
{context}

Latest question:
{state["question"]}
"""

    response = model.invoke(prompt)
    answer = response.content.encode("ascii", errors="ignore").decode()

    return {"answer": answer}


def route_after_clarity(state: HRState):
    # Route after clarity check
    if state["needs_clarification"]:
        return "ask_question"
    return "classify_topic"


def build_graph():
    # Build the LangGraph workflow
    graph = StateGraph(HRState)

    graph.add_node("check_clarity", check_clarity)
    graph.add_node("ask_question", ask_question)
    graph.add_node("classify_topic", classify_topic)
    graph.add_node("answer_by_topic", answer_by_topic)

    graph.add_edge(START, "check_clarity")
    graph.add_conditional_edges("check_clarity", route_after_clarity)
    graph.add_edge("classify_topic", "answer_by_topic")
    graph.add_edge("ask_question", END)
    graph.add_edge("answer_by_topic", END)

    return graph


def create_app():
    # Compile the graph
    graph = build_graph()
    return graph.compile()


def main():
    # Run the interactive loop
    app = create_app()

    history = ""
    pending_question = None
    pending_clarifier = None

    while True:
        user_input = input("\nEnter your question (or type exit): ").strip()

        if user_input.lower() == "exit":
            print("Goodbye.")
            break

        if pending_question:
            question = f"""
Original question:
{pending_question}

Clarifying question:
{pending_clarifier}

User clarification:
{user_input}
"""
            pending_question = None
            pending_clarifier = None
        else:
            question = user_input

        result = app.invoke({
            "question": question,
            "history": history,
            "needs_clarification": False,
            "clarifying_question": "",
            "topic": "",
            "answer": ""
        })

        if result["answer"].startswith("CLARIFY:"):
            clarifier = result["answer"].replace("CLARIFY:", "", 1).strip()
            print("\n=== CLARIFYING QUESTION ===")
            print(clarifier)

            pending_question = user_input
            pending_clarifier = clarifier
            history += f"\nUser: {user_input}\nAssistant: {clarifier}"
        else:
            print("\n=== FINAL ANSWER ===")
            print(result["answer"])
            history += f"\nUser: {user_input}\nAssistant: {result['answer']}"


if __name__ == "__main__":
    main()
