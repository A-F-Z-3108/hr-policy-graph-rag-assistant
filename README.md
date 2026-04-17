# HR Policy Graph RAG Assistant

## Overview
This project is a graph-based HR policy assistant that answers questions using real policy documents and retrieval-augmented generation (RAG). It uses a small conversation workflow instead of a single fixed response.

The assistant can:
- read multiple HR-related PDF documents
- build a FAISS vector index from document chunks
- retrieve relevant passages for a user question
- answer from retrieved context only
- ask a clarifying question when the user's request is too vague
- route the question into one of several topic branches:
  - leave / FMLA
  - ADA / reasonable accommodation
  - OSHA / worker rights
- show a simple graph trace for debugging and demo purposes

## Main Files
- `build_index.py`  
  Reads the PDF files, splits them into chunks, creates embeddings, builds a FAISS index, and saves it locally.

- `hr_graph.py`  
  Main application. Loads the saved FAISS index, handles user questions, asks clarifying questions when needed, routes by topic, retrieves relevant context, and generates answers.

- `data/`  
  Contains the source PDF files used as the knowledge base.

- `faiss_index/`  
  Contains the saved FAISS index generated from the PDF documents.

- `archive/`  
  Contains earlier experimental and testing scripts that are not required for the final demo.

## How It Works
1. PDF documents are stored in the `data/` folder.
2. `build_index.py` reads the PDFs and splits them into chunks.
3. Embeddings are created for the chunks.
4. A FAISS index is built and saved in `faiss_index/`.
5. `hr_graph.py` loads the saved index and starts the interactive assistant.
6. The assistant checks whether the question is clear enough.
7. If needed, it asks a clarifying question.
8. If the question is clear, it routes it to the correct topic branch.
9. It retrieves relevant chunks and answers only from the retrieved context.

## Supported Topic Branches
- Leave / FMLA
- ADA / reasonable accommodation
- OSHA / workplace safety / worker rights

## Example Questions
- Can an employee take leave to care for a parent with a serious health condition?
- Does an employer have to provide a reasonable accommodation for an employee with a disability?
- What are an employer's responsibilities after a federal OSHA inspection?
- What about leave?

## Requirements
Typical packages used in this project:
- langchain
- langchain-openai
- langchain-community
- langgraph
- pypdf
- faiss-cpu

## Running the Project

### 1. Build the index
```bash
python build_index.py
