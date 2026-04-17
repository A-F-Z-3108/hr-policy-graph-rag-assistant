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
    route: str

index_dir = "faiss_index"

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = FAISS.load_local(
    index_dir,
    embeddings,
    allow_dangerous_deserialization=True
)

model = ChatOpenAI(model="gpt-5.4")

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
    question = state["question"]

    if "User clarification:" in question:
        return {
            "needs_clarification": False,
            "clarifying_question": "",
            "route": "clear"
        }

    if rule_based_clarity(question):
        return {
            "needs_clarification": False,
            "clarifying_question": "",
            "route": "clear"
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
        "clarifying_question": response.content.strip(),
        "route": "clarify"
    }

def ask_question(state: HRState):
    return {
        "answer": f"CLARIFY: {state['clarifying_question']}",
        "route": "ask_question"
    }

def classify_topic(state: HRState):
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
        return {"topic": "osha", "route": "topic_osha"}

    if any(x in q for x in [
        "accommodation",
        "reasonable accommodation",
        "disability",
        "disabled",
        "ada",
        "medical condition"
    ]):
        return {"topic": "ada", "route": "topic_ada"}

    return {"topic": "leave", "route": "topic_leave"}

def build_context(question, history, topic):
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
        source = (doc.metadata.get("source_file") or "").lower()

        if topic == "leave" and (
            "family and medical leave" in source or
            "leave act" in source or
            "fmla" in source
        ):
            filtered_docs.append(doc)

        elif topic == "ada" and (
            "accommodation" in source or
            "disabilities act" in source or
            "ada" in source
        ):
            filtered_docs.append(doc)

        elif topic == "osha" and (
            "osha" in source or
            "inspection" in source or
            "workers’ rights" in source or
            "workers' rights" in source
        ):
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

    return {
        "answer": answer,
        "route": f"answer_{state['topic']}"
    }

def route_after_clarity(state: HRState):
    if state["needs_clarification"]:
        return "ask_question"
    return "classify_topic"

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

app = graph.compile()

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
        "answer": "",
        "route": ""
    })

    print("\n=== GRAPH TRACE ===")
    print("needs_clarification:", result["needs_clarification"])
    print("topic:", result["topic"])
    print("route:", result["route"])

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
