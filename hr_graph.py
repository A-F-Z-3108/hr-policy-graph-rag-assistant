from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS


class HRState(TypedDict):
    question: str
    history: str
    needs_clarification: bool
    clarifying_question: str
    answer: str


# Create embeddings
def get_embeddings():
    return OpenAIEmbeddings(model="text-embedding-3-small")


# Load FAISS index
def get_vectorstore():
    return FAISS.load_local(
        "faiss_index",
        get_embeddings(),
        allow_dangerous_deserialization=True
    )


# Create chat model
def get_model():
    return ChatOpenAI(model="gpt-5.4")


# Check whether the question is clear enough
def rule_based_clarity(question: str):
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

    clear_keywords = [
        "fmla", "leave", "absence",
        "accommodation", "reasonable accommodation", "disability", "ada",
        "osha", "inspection", "worker rights", "safety", "hazard"
    ]

    if any(k in q for k in clear_keywords):
        return True

    if len(q.split()) >= 8:
        return True

    return False


# Decide whether to ask for clarification
def check_clarity(state: HRState):
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
    response = get_model().invoke(prompt)

    return {
        "needs_clarification": True,
        "clarifying_question": response.content.strip()
    }


# Return the clarifying question
def ask_question(state: HRState):
    return {"answer": f"CLARIFY: {state['clarifying_question']}"}


# Build retrieved context
def build_context(question, history):
    retrieval_query = question
    if history:
        retrieval_query = f"""
Conversation so far:
{history}

Latest question:
{question}
"""

    docs = get_vectorstore().similarity_search(retrieval_query, k=4)

    context_parts = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source_file")
        page = doc.metadata.get("page")
        context_parts.append(
            f"[Source {i}: {source}, page {page}]\n{doc.page_content}"
        )

    return "\n\n".join(context_parts)


# Generate the final answer
def answer_question(state: HRState):
    context = build_context(state["question"], state["history"])

    prompt = f"""
You are an HR policy assistant.

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

    response = get_model().invoke(prompt)
    return {"answer": response.content.encode("ascii", errors="ignore").decode()}


# Route after clarity check
def route_after_clarity(state: HRState):
    if state["needs_clarification"]:
        return "ask_question"
    return "answer_question"


# Build the graph
def build_graph():
    graph = StateGraph(HRState)

    graph.add_node("check_clarity", check_clarity)
    graph.add_node("ask_question", ask_question)
    graph.add_node("answer_question", answer_question)

    graph.add_edge(START, "check_clarity")
    graph.add_conditional_edges("check_clarity", route_after_clarity)
    graph.add_edge("ask_question", END)
    graph.add_edge("answer_question", END)

    return graph.compile()


# Run the app
def main():
    app = build_graph()

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
