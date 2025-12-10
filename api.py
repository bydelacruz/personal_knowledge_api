"""
API layer for managing note entries using FastAPI.
"""

import os
import shutil  # for saving files
from contextlib import asynccontextmanager

import chromadb
import google.generativeai as genai
from fastapi import Depends, FastAPI, File, UploadFile
from pydantic import BaseModel
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

# import internal logic
from database import NoteRepository
from ingestion import chunk_text, extract_text_from_pdf
from models import NoteEntry


# --- 1. THE PYDANTIC SCHEMAS (The Data Contract) ---
# We define what the API *expects* from the user.
# Notice we dont include 'id' here, because the user doesn't create IDs.
class NoteCreatePayload(BaseModel):
    topic: str
    tags: list[str]
    rating: int


# We define what the API *returns* to the user.
# Here, 'id' is required because the DB has generated it.
class NoteResponse(BaseModel):
    id: int
    topic: str
    tags: list[str]
    rating: int
    is_high_priority: bool  # We will compute this on the fly


class SearchResponse(BaseModel):  # for use with ChromaDB
    id: int
    topic: str
    rating: int
    similarity_score: float


class ChatRequest(BaseModel):
    question: str


class ChatResponse(BaseModel):
    answer: str
    sources: list[str]  # Transparency: Show user where we found the info


# --- CONFIGURATION ---
IS_RENDER = os.environ.get("RENDER", False)
BASE_DIR = "/var/data" if IS_RENDER else "."


# --- GLOBAL VARIABLES ---
# we need to hold the chroma client in memory so we don't reload it every request.
chroma_client = None
note_collection = None
bm25_index: BM25Okapi = None
bm25_text_map: dict[int, str] = {}  # Maps SQLite ID to the raw text chunk
reranker_model = None


# --- SEED DATA ---
# This ensures portfolio is NEVER empty
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
    {
        "topic": "Ben's Goal: To build scalable, intelligent systems that solve real-world problems.",
        "tags": ["goals", "portfolio"],
        "rating": 8,
    },
]


# --- LIFESPAN MANAGER ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    # 1. Setup DB
    repo = NoteRepository(f"{BASE_DIR}/notes.db")
    await repo.create_table()

    # 2. Setup AI Components
    global chroma_client, note_collection, bm25_index, bm25_text_map, reranker_model
    chroma_client = chromadb.PersistentClient(path=f"{BASE_DIR}/my_vector_db")
    note_collection = chroma_client.get_or_create_collection(name="my_notes")

    print("Startup: Loading Reranker (this takes a moment)...")
    reranker_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    # 3. CHECK FOR EMPTY DB & SEED IT
    # If the database is empty (which happens on Render Free Tier restart),
    # we auto-populate it so the recruiter sees something cool immediately.
    existing_notes = await repo.get_all_notes()
    if not existing_notes:
        print("Startup: Database empty. Seeding default portfolio data...")
        for note_data in SEED_DATA:
            new_entry = NoteEntry(
                topic=note_data["topic"],
                tags=note_data["tags"],
                rating=note_data["rating"],
            )
            created_id = await repo.add_note(new_entry)
            note_collection.add(
                documents=[note_data["topic"]],
                metadatas=[{"rating": note_data["rating"]}],
                ids=[str(created_id)],
            )
        # Re-fetch for BM25
        existing_notes = await repo.get_all_notes()

    # 4. Build BM25
    print("Startup: Building Keyword Index...")
    corpus = [note.topic for note in existing_notes]
    if corpus:
        tokenized_corpus = [doc.split(" ") for doc in corpus]
        bm25_index = BM25Okapi(tokenized_corpus)
        bm25_text_map = {note.id: note.topic for note in existing_notes}
    else:
        # Handle empty case just in case
        bm25_index = None
        bm25_text_map = {}

    print("Startup: System Ready.")
    yield
    print("Shutdown: Cleanup complete.")


# helper function for updating the BM25 index
async def update_bm25_index(note_id: int, text_content: str):
    global bm25_index, bm25_text_map

    # 1. Update the map with the new entry
    bm25_text_map[note_id] = text_content

    # 2. Rebuild the index (BM25 doesn't easily support incremental updates)
    corpus = list(bm25_text_map.values())
    tokenized_corpus = [doc.split(" ") for doc in corpus]
    bm25_index = BM25Okapi(tokenized_corpus)

    print(f"BM25 Index updated with document ID {note_id}.")


# helper function that performs both searches, combines the result
# and returns a unique set of IDs
def run_hybrid_search(query: str, n_results: int = 10) -> list[str]:
    # 1. BM25 (keyword) search
    tokenized_query = query.split(" ")
    doc_scores = bm25_index.get_scores(tokenized_query)

    # Rank indices based on scores
    # We get the top N indices from the scores array
    ranked_indices = doc_scores.argsort()[-n_results:][::-1]

    bm25_results = []
    # Get the actual IDs from the ranked indices
    id_list = list(bm25_text_map.keys())
    for index in ranked_indices:
        if index < len(id_list):
            bm25_results.append(str(id_list[index]))

    # 2. chroma (Vector) Search
    chroma_results = note_collection.query(query_texts=[query], n_results=n_results)

    chroma_ids = chroma_results["ids"][0] if chroma_results["ids"] else []

    # 3. combine and Deduplicate
    # We cobine the lists and use a set to ensure every ID is unique
    combined_ids = list(set(bm25_results + chroma_ids))

    return combined_ids


# Helper function that reranks documents
def rerank_documents(query: str, candidate_ids: list[str], top_k: int = 5) -> list[str]:
    """
    1. Fetches text for candidate IDs.
    2. Pairs them with the query.
    3. Scores them using the Cross-Encoder.
    4. Returns the top_k best text chunks.
    """
    # 1. Fetch Text
    # We use the bm25_text_map because it's a fast ID->Text lookup
    candidates_text = []
    valid_ids = []

    for doc_id_str in candidate_ids:
        doc_id = int(doc_id_str)
        if doc_id in bm25_text_map:
            text = bm25_text_map[doc_id]
            candidates_text.append(text)
            valid_ids.append(doc_id)

    if not candidates_text:
        return []

    # 2. Prepare Pairs for the Model
    # Format: [[Query, Doc1], [Query, Doc2], ...]
    pairs = [[query, doc] for doc in candidates_text]

    # 3. Score
    scores = reranker_model.predict(pairs)

    # 4. Sort and Filter
    # We zip the scores with the text so we keep them together
    scored_results = list(zip(scores, candidates_text))

    # Sort by score (descending)
    scored_results.sort(key=lambda x: x[0], reverse=True)

    # Return just the text of the top k winners
    top_results = [text for score, text in scored_results[:top_k]]

    return top_results


# --- 2. THE APP SETUP ---
# We pass the lifespan manager to FastAPI
app = FastAPI(title="Personal Knowledge API", version="1.0.0", lifespan=lifespan)


# --- 3. DEPENDENCY INJECTION ---
# Instead of creating a new repository manually in every function,
# we define this helper. FastAPI will call this, cache the result if needed,
# and pass it to our routes. This makes testing infinitely easier later.
def get_repository():
    """
    Dependency injector for NoteRepository.
    """
    # In a real massive app, we might grab DB credentials from env vars here.
    return NoteRepository("notes.db")


# --- 4. THE ROUTES (The Menu) ---


@app.post("/notes", response_model=NoteResponse, status_code=201)
async def create_note(
    payload: NoteCreatePayload, repo: NoteRepository = Depends(get_repository)
):
    """
    Creates a new note.
    1. Validates input via NoteCreatePayload.
    2. Converts Pydantic payload -> Domain Object (NoteEntry)
    3. Saves to DB via repository
    4. Returns the created object
    """
    # convert Pydantic -> Internal Domain Model
    new_entry = NoteEntry(topic=payload.topic, tags=payload.tags, rating=payload.rating)

    # Save to DB
    created_id = await repo.add_note(new_entry)

    # Save to ChromaDB (The seaching index)
    # we use the SQLite ID as the Chroma ID so we can link them later.
    note_collection.add(
        documents=[payload.topic],  # the text to vectorize
        metadatas=[{"rating": payload.rating}],  # extra data
        ids=[str(created_id)],  # must be string for chroma
    )

    # Update keyword index
    await update_bm25_index(created_id, payload.topic)

    return NoteResponse(
        id=created_id,
        topic=new_entry.topic,
        tags=new_entry.tags,
        rating=new_entry.rating,
        is_high_priority=new_entry.is_high_priority(),
    )


@app.get("/notes", response_model=list[NoteResponse])
async def get_notes(repo: NoteRepository = Depends(get_repository)):
    """
    Fetches all notes.
    FastAPI automatically handles the conversion from our NoteEntry objects
    to the NoteResponse JSON format.
    """
    domain_objects = await repo.get_all_notes()

    # We map our internal onjects to the API response format
    results = []
    for note in domain_objects:
        results.append(
            NoteResponse(
                id=note.id,
                topic=note.topic,
                tags=note.tags,
                rating=note.rating,
                is_high_priority=note.is_high_priority(),
            )
        )
    return results


@app.get("/search", response_model=list[SearchResponse])
async def search_notes(query: str, repo: NoteRepository = Depends(get_repository)):
    """
    Semantic Search:
    1. Takes user query
    2. Uses Chroma to find the best matching IDs.
    3. (Optional) Could fetch full details from SQLite if needed.
    """
    # Ask chroma for the top 5 matches
    results = note_collection.query(
        query_texts=[query],
        n_results=5,
    )

    # parse results
    # results['ids'][10] is a list of IDs ['1', '5', '2']
    # results['distances'][10] is a list of scores
    found_notes = []

    # check if we found anyting
    if not results["ids"][0]:
        return []  # empty list if no matches

    count = len(results["ids"][0])

    for i in range(count):
        note_id_str = results["ids"][0][i]
        # in Chroma, lower distance = better match
        score = results["distances"][0][i]
        # however, typically cosine distance: 0 is perfect, 1 is opposite
        # let's just return what chroma gives us for now

        # we also get the metadata back
        meta = results["metadatas"][0][i]
        doc_text = results["documents"][0][i]

        found_notes.append(
            SearchResponse(
                id=int(note_id_str),
                topic=doc_text,
                rating=meta.get("rating", 0),
                similarity_score=score,
            )
        )

    return found_notes


# --- 🚀 THE GEMINI RAG ENDPOINT ---
@app.post("/ask", response_model=ChatResponse)
async def ask_question(request: ChatRequest):
    # 1. RETRIEVAL
    candidate_ids = run_hybrid_search(request.question, n_results=20)

    if not candidate_ids:
        return ChatResponse(answer="I don't have information on that.", sources=[])

    # 2. RERANKING
    top_docs = rerank_documents(request.question, candidate_ids, top_k=5)

    if not top_docs:
        return ChatResponse(answer="I found notes, but none seem relevant.", sources=[])

    # 3. GENERATION (Gemini Swap)
    context_text = "\n".join(top_docs)

    # Check for API Key
    api_key = os.environ.get("GEMINI_API_KEY")

    if api_key:
        try:
            # Configure Gemini
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel("gemini-1.5-flash")  # Fast & Free

            prompt = f"""
            You are a helpful assistant. Answer the user's question based ONLY on the following context.
            If the answer is not in the context, say "I don't know."

            Context:
            {context_text}

            User Question: {request.question}
            """

            response = model.generate_content(prompt)
            final_answer = response.text

        except Exception as e:
            final_answer = f"Error calling Gemini: {str(e)}"
    else:
        final_answer = "Error: GEMINI_API_KEY not set on server."

    return ChatResponse(answer=final_answer, sources=top_docs)


@app.post("/ask", response_model=ChatResponse)
# --- THE INGESTION ENDPOINT ---
@app.post("/upload-pdf")
async def upload_pdf(
    file: UploadFile = File(...), repo: NoteRepository = Depends(get_repository)
):
    """
    1. Save the PDF to disk.
    2. Extracts text.
    3. Chunks it.
    4. Saves each chunk to the DB and Vector Index.
    """
    # 1. Save the file locally so pypdf can read it
    file_location = f"uploads/{file.filename}"

    # shutil.copyfileobj is a high-speed tool to copy data streams
    with open(file_location, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # 2. Extract text (The 'ingestion.py' logic)
    print(f"Processing {file.filename}...")
    raw_text = extract_text_from_pdf(file_location)

    # 3. Chunk Text
    chunks = chunk_text(raw_text, chunk_size=500, overlap=50)

    print(f"Split into {len(chunks)} chunks.")

    # 4. Save each chunk (Looping through the list)
    saved_count = 0
    for i, chunk_text_content in enumerate(chunks):
        # Create a descriptive title
        chunk_title = f"{file.filename} - Chunk {i + 1}"

        # Create the Domain Object
        new_entry = NoteEntry(
            topic=chunk_text_content,
            tags=["pdf_import", file.filename],  # Tag it so we know source
            rating=5,  # Default rating for imported data
        )

        # Save to SQLite
        created_id = await repo.add_note(new_entry)

        # Save to Chroma (The Brain)
        note_collection.add(
            documents=[chunk_text_content],
            metadatas=[{"rating": 5, "source": file.filename}],
            ids=[str(created_id)],
        )

        # Update keyword index
        await update_bm25_index(created_id, chunk_text_content)
        saved_count += 1

    # 5. cleanup (Optional: Delete the file after processing to save space)
    os.remove(file_location)

    return {
        "filename": file.filename,
        "chunks_processed": saved_count,
        "status": "success",
    }


# health check (Always good practice)
@app.get("/health")
async def health_check():
    """
    Simple health check endpoint.
    """
    return {"status": "operational", "db_connected": True}
    return {"status": "operational", "db_connected": True}
