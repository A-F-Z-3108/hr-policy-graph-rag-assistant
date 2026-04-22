# HR Policy Graph RAG Assistant

## Overview
This project is a lightweight graph-based HR policy assistant built on top of retrieval-augmented generation (RAG). It uses real PDF policy documents as its knowledge base and answers user questions from retrieved document context only.

The assistant can:
- read multiple HR-related policy documents
- retrieve relevant document chunks from a FAISS index
- answer policy questions from retrieved context
- ask a clarifying question when the user query is too vague
- maintain short conversation history across turns
- use a minimal LangGraph workflow for flow control

## Main Idea
The project combines:
- **RAG** for document-grounded answers
- **FAISS** for vector search
- **OpenAI embeddings** for semantic retrieval
- **ChatOpenAI** for answer generation
- **LangGraph** for simple conversational routing

The graph is intentionally minimal. It controls the flow between:
1. clarity check
2. optional clarification
3. final answer generation

## Project Structure
- `build_index.py`  
  Reads the PDF files from `data/`, splits them into chunks, creates embeddings, builds a FAISS index, and saves it locally.

- `hr_graph.py`  
  Main application. Loads the saved FAISS index, checks whether the user question is clear enough, optionally asks a clarifying question, retrieves relevant context, and generates a final answer.

- `data/`  
  Source PDF files used as the knowledge base.

- `faiss_index/`  
  Saved FAISS vector index generated from the source documents.

- `archive/`  
  Older experimental and intermediate scripts that are not required for the final version.

## How It Works
1. Policy PDFs are stored in the `data/` folder.
2. `build_index.py` reads the files and splits them into chunks.
3. Embeddings are generated for the chunks.
4. A FAISS index is created and stored locally.
5. `hr_graph.py` loads the index and starts an interactive session.
6. The graph checks whether the user question is clear.
7. If needed, it asks one clarifying question.
8. If the question is clear, it retrieves relevant chunks.
9. The model answers only from the retrieved context.

## Graph Flow
The LangGraph workflow is minimal and used only for flow control:

`START -> check_clarity -> ask_question OR answer_question -> END`

This keeps the architecture simple while still demonstrating graph-based orchestration.

## Features
- Multi-document RAG over real HR-related PDF files
- Clarification loop for vague questions
- Retrieval grounded in source documents
- Conversation history support
- Simple graph-based workflow
- Source-aware prompting with file name and page number

## Example Questions
- Can an employee take leave to care for a parent with a serious health condition?
- How many weeks of unpaid leave can an eligible employee take under FMLA?
- Does an employer have to provide a reasonable accommodation for an employee with a disability?
- What are an employer's responsibilities after a federal OSHA inspection?
- What about leave?

## Requirements
Typical packages used in this project:
- `langchain`
- `langchain-openai`
- `langchain-community`
- `langgraph`
- `pypdf`
- `faiss-cpu`

## Setup

### 1. Install dependencies
```bash
pip install -U langchain langchain-openai langchain-community langgraph pypdf faiss-cpu
