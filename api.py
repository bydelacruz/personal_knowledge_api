"""
API layer for managing note entries using FastAPI.
"""

import os
import shutil
from contextlib import asynccontextmanager

import chromadb
import google.generativeai as genai
from chromadb.utils import embedding_functions  # <--- NEW
from fastapi import Depends, FastAPI, File, UploadFile
from pydantic import BaseModel
from rank_bm25 import BM25Okapi

from database import NoteRepository

# Import ingestion logic
from ingestion import chunk_text, extract_text_from_pdf
from models import NoteEntry

# --- CONFIGURATION ---
IS_RENDER = os.environ.get("RENDER", False)
BASE_DIR = "."
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

# --- GLOBAL VARIABLES ---
chroma_client = None
note_collection = None
bm25_index = None
bm25_text_map = {}


# --- CUSTOM GEMINI EMBEDDING FUNCTION ---
# This is the secret sauce. It runs on Google's servers, not your RAM.
class GeminiEmbeddingFunction(embedding_functions.EmbeddingFunction):
    def __call__(self, input: list[str]) -> list[list[float]]:
        if not GEMINI_API_KEY:
            return []
        genai.configure(api_key=GEMINI_API_KEY)
        model = "models/text-embedding-004"

        embeddings = []
        for text in input:
            # Google's API returns the vector
            result = genai.embed_content(model=model, content=text)
            embeddings.append(result["embedding"])
        return embeddings


# --- SEED DATA ---
SEED_DATA = [
    {
        "topic": "About Ben: Ben is a Backend Engineer who specializes in Python, FastAPI, and AI integration.",
        "tags": ["bio", "portfolio"],
        "rating": 10,
    },
    {
        "topic": "Ben's Tech Stack: He uses Python 3.11, Docker, AWS, and RAG architectures.",
        "tags": ["skills", "portfolio"],
        "rating": 9,
    },
]


# --- LIFESPAN MANAGER ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    # 1. Setup DB
    repo = NoteRepository(f"{BASE_DIR}/notes.db")
    await repo.create_table()

    # 2. Setup Chroma with Google's Brain (Not Local RAM)
    global chroma_client, note_collection, bm25_index, bm25_text_map
    chroma_client = chromadb.PersistentClient(path=f"{BASE_DIR}/my_vector_db")

    # We explicitly tell Chroma to use our custom function
    gemini_ef = GeminiEmbeddingFunction()

    note_collection = chroma_client.get_or_create_collection(
        name="my_notes",
        embedding_function=gemini_ef,
    )

    # 3. SEED DATA
    existing_notes = await repo.get_all_notes()
    if not existing_notes:
        print("Startup: Seeding database...")
        for note_data in SEED_DATA:
            new_entry = NoteEntry(
                topic=note_data["topic"],
                tags=note_data["tags"],
                rating=note_data["rating"],
            )
            created_id = await repo.add_note(new_entry)

            # Chroma now calls Google API to get vectors. No RAM used!
            note_collection.add(
                documents=[note_data["topic"]],
                metadatas=[{"rating": note_data["rating"]}],
                ids=[str(created_id)],
            )
        existing_notes = await repo.get_all_notes()

    # 4. Build BM25
    corpus = [note.topic for note in existing_notes]
    if corpus:
        tokenized_corpus = [doc.split(" ") for doc in corpus]
        bm25_index = BM25Okapi(tokenized_corpus)
        bm25_text_map = {note.id: note.topic for note in existing_notes}
    else:
        bm25_index = None
        bm25_text_map = {}

    print("Startup: System Ready.")
    yield
    print("Shutdown: Cleanup complete.")


app = FastAPI(title="Personal Knowledge Brain", lifespan=lifespan)


def get_repository():
    return NoteRepository(f"{BASE_DIR}/notes.db")


# --- MODELS ---
class NoteCreatePayload(BaseModel):
    topic: str
    tags: list[str]
    rating: int


class NoteResponse(BaseModel):
    id: int
    topic: str
    tags: list[str]
    rating: int
    is_high_priority: bool


class ChatRequest(BaseModel):
    question: str


class ChatResponse(BaseModel):
    answer: str
    sources: list[str]


# --- ROUTES ---


@app.post("/notes", response_model=NoteResponse, status_code=201)
async def create_note(
    payload: NoteCreatePayload, repo: NoteRepository = Depends(get_repository)
):
    new_entry = NoteEntry(topic=payload.topic, tags=payload.tags, rating=payload.rating)
    created_id = await repo.add_note(new_entry)

    # Save to Chroma (Calls Google API)
    note_collection.add(
        documents=[payload.topic],
        metadatas=[{"rating": payload.rating}],
        ids=[str(created_id)],
    )

    # Update BM25
    global bm25_index, bm25_text_map
    bm25_text_map[created_id] = payload.topic
    corpus = list(bm25_text_map.values())
    tokenized_corpus = [doc.split(" ") for doc in corpus]
    bm25_index = BM25Okapi(tokenized_corpus)

    return NoteResponse(
        id=created_id,
        topic=new_entry.topic,
        tags=new_entry.tags,
        rating=new_entry.rating,
        is_high_priority=new_entry.is_high_priority(),
    )


@app.post("/upload-pdf")
async def upload_pdf(
    file: UploadFile = File(...), repo: NoteRepository = Depends(get_repository)
):
    file_location = f"uploads/{file.filename}"
    with open(file_location, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    raw_text = extract_text_from_pdf(file_location)
    chunks = chunk_text(
        raw_text, chunk_size=1000, overlap=100
    )  # Bigger chunks for API efficiency

    saved_count = 0
    for i, chunk_text_content in enumerate(chunks):
        new_entry = NoteEntry(
            topic=chunk_text_content, tags=["pdf_import", file.filename], rating=5
        )
        created_id = await repo.add_note(new_entry)

        # Calls Google API
        note_collection.add(
            documents=[chunk_text_content],
            metadatas=[{"rating": 5, "source": file.filename}],
            ids=[str(created_id)],
        )

        # Update BM25 map
        bm25_text_map[created_id] = chunk_text_content
        saved_count += 1

    # Rebuild BM25 once at the end of upload (Efficiency)
    global bm25_index
    corpus = list(bm25_text_map.values())
    tokenized_corpus = [doc.split(" ") for doc in corpus]
    bm25_index = BM25Okapi(tokenized_corpus)

    return {
        "filename": file.filename,
        "chunks_processed": saved_count,
        "status": "success",
    }


@app.post("/ask", response_model=ChatResponse)
async def ask_question(request: ChatRequest):
    # 1. RETRIEVAL (Hybrid: BM25 + Chroma)
    # Chroma now uses Google Vectors to find matches.

    # BM25 Search
    tokenized_query = request.question.split(" ")
    if bm25_index:
        doc_scores = bm25_index.get_scores(tokenized_query)
        ranked_indices = doc_scores.argsort()[-5:][::-1]
        bm25_ids = [
            str(list(bm25_text_map.keys())[i])
            for i in ranked_indices
            if i < len(bm25_text_map)
        ]
    else:
        bm25_ids = []

    # Vector Search (Google)
    chroma_results = note_collection.query(query_texts=[request.question], n_results=5)
    chroma_ids = chroma_results["ids"][0] if chroma_results["ids"] else []

    # Combine
    combined_ids = list(set(bm25_ids + chroma_ids))

    # Fetch Text
    top_docs = []
    for doc_id_str in combined_ids:
        doc_id = int(doc_id_str)
        if doc_id in bm25_text_map:
            top_docs.append(bm25_text_map[doc_id])

    if not top_docs:
        return ChatResponse(answer="I don't have information on that.", sources=[])

    # 2. GENERATION (Gemini)
    context_text = "\n".join(top_docs)

    if GEMINI_API_KEY:
        try:
            genai.configure(api_key=GEMINI_API_KEY)
            model = genai.GenerativeModel("gemini-flash-latest")
            prompt = f"""Context:\n{context_text}\n\nQuestion: {
                request.question
            }\nAnswer based on context only:"""
            response = model.generate_content(prompt)
            final_answer = response.text
        except Exception as e:
            final_answer = f"Error: {str(e)}"
    else:
        final_answer = "Error: API Key missing."

    return ChatResponse(answer=final_answer, sources=top_docs)


# health check (Always good practice)
@app.get("/health")
async def health_check():
    """
    Simple health check endpoint.
    """
    return {"status": "operational", "db_connected": True}
    return {"status": "operational", "db_connected": True}
