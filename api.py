"""
API layer for managing note entries using FastAPI.
"""

from fastapi import Depends, FastAPI
from pydantic import BaseModel

# import internal logic
from database import NoteRepository
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


# --- 2. THE APP SETUP ---
app = FastAPI(title="Personal Knowledge API", version="1.0.0")

# --- 3. DEPENDENCY INJECTION (The wizardry) ---
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
def create_note(
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
    created_id = repo.add_note(new_entry)

    return NoteResponse(
        id=created_id,
        topic=new_entry.topic,
        tags=new_entry.tags,
        rating=new_entry.rating,
        is_high_priority=new_entry.is_high_priority(),
    )


@app.get("/notes", response_model=list[NoteResponse])
def get_notes(repo: NoteRepository = Depends(get_repository)):
    """
    Fetches all notes.
    FastAPI automatically handles the conversion from our NoteEntry objects
    to the NoteResponse JSON format.
    """
    domain_objects = repo.get_all_notes()

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


# health check (Always good practice)
@app.get("/health")
def health_check():
    """
    Simple health check endpoint.
    """
    return {"status": "operational", "db_connected": True}
