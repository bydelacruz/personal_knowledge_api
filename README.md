# 🧠 Personal Knowledge Brain (AI-Powered RAG)

A full-stack, retrieval-augmented generation (RAG) system that allows users to perform **semantic searches** and **chat** with their private documents (PDFs, Notes).

Built with **Python 3.11**, **FastAPI**, and **Streamlit**, and containerized with **Docker**.

---

## 🏗️ Architecture

This project implements a **Hybrid Search RAG Pipeline** designed for zero-cost deployment while maintaining high accuracy.

1.  **Ingestion Layer:**
    - **PDF Parsing:** Uses `pypdf` to extract text.
    - **Chunking:** Implements a sliding window algorithm (500 chars, 50 overlap) to preserve semantic context.
    - **Vectorization:** Google Gemini (`text-embedding-004`) converts text into high-dimensional vectors.

2.  **Retrieval Layer (Hybrid Search):**
    - **Semantic Search:** Uses **ChromaDB** to find conceptual matches.
    - **Keyword Search:** Uses **BM25** (Best Match 25) to catch exact keywords that vector search might miss.
    - **Result Merging:** Combines and deduplicates results from both indices for maximum recall.

3.  **Generation Layer:**
    - The top retrieved contexts are injected into a system prompt.
    - **Google Gemini 1.5 Flash** generates the final answer, grounded strictly in the provided data to prevent hallucinations.

4.  **Infrastructure:**
    - **Asynchronous API:** Built on `FastAPI` + `uvicorn` for high concurrency.
    - **Containerization:** Custom `Dockerfile` managing a multi-process entrypoint (API + Streamlit).
    - **Self-Healing Database:** Implements an auto-seeding script on startup to handle ephemeral file systems on free-tier cloud hosting.

---

## ⚡ Features

- **📄 PDF Ingestion:** Upload any PDF; it is automatically chunked, vectorized, and indexed.
- **💬 Context-Aware Chat:** Ask questions like "What is the email in that resume?" and get precise answers.
- **🔎 Hybrid Search:** Combines the power of Vectors (meaning) and BM25 (keywords).
- **🛡️ Source Transparency:** The AI cites the exact text chunks used to generate the answer.
- **🧠 Zero-Cost Architecture:** Runs entirely on free tiers (Render + Google Gemini API) without needing GPU servers.

---

## 🛠️ Tech Stack

- **Language:** Python 3.11 (Optimized for compatibility)
- **Backend:** FastAPI, Uvicorn, Aiosqlite
- **Frontend:** Streamlit
- **Database:** ChromaDB (Vector), SQLite (Relational)
- **AI/ML:** Google Gemini API, Rank-BM25
- **DevOps:** Docker, GitHub Actions (optional), Render

---

## 🚀 Local Setup

**Prerequisites:** Python 3.11+, Gemini API Key.

1.  **Clone the Repository**

    ```bash
    git clone [https://github.com/yourusername/personal-knowledge-brain.git](https://github.com/yourusername/personal-knowledge-brain.git)
    cd personal-knowledge-brain
    ```

2.  **Create Virtual Environment**

    ```bash
    python3.11 -m venv venv
    source venv/bin/activate
    ```

3.  **Install Dependencies**

    ```bash
    pip install -r requirements.txt
    ```

4.  **Set Environment Variables**

    ```bash
    export GEMINI_API_KEY="your_google_api_key_here"
    ```

5.  **Run the System**

    ```bash
    # Terminal 1: Backend
    uvicorn api:app --reload

    # Terminal 2: Frontend
    streamlit run app.py
    ```

---

## ☁️ Deployment (Docker)

This application is configured for deployment on **Render (Free Tier)**.

1.  **Docker Build:** The `Dockerfile` handles the installation of system dependencies.
2.  **Entrypoint:** `entrypoint.sh` manages the concurrent startup of the FastAPI backend and Streamlit frontend within a single container.
3.  **Environment Variables:**
    - `GEMINI_API_KEY`: Required for Embeddings and Generation.
    - `PYTHON_VERSION`: Set to `3.11.9`.

---

## 🔮 Future Improvements

- **Reranking:** Implement a Cross-Encoder (e.g., `ms-marco`) to re-score the top 20 results for higher precision (requires >1GB RAM).
- **Auth:** Add API Key authentication for private routes.
- **Cloud Database:** Migration from SQLite/Chroma local files to Postgres/Pinecone for true persistence across restarts.

---

### 🧙‍♂️ A Note from the Engineer

This project demonstrates how to overcome the "Cold Start" problem in ephemeral environments by using an intelligent seeding script, allowing for a fully functional, impressive AI demo that costs **$0.00/month** to host.
