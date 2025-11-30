"""
A simple note-taking application that demonstrates
the repository pattern for database interactions.
"""

from database import NoteRepository
from models import NoteEntry

if __name__ == "__main__":
    # 1. setup db
    repo = NoteRepository()

    # 2. create a python object (business logic)
    my_note = NoteEntry(
        topic="Python Generators", tags=["coding", "advanced"], rating=9
    )

    # 3. save the object to the db
    # the repo handles the conversion to a db-friendly format
    print(f"Saving note: {my_note.topic}")
    repo.add_note(my_note)

    # 4. retrieve and display all notes
    saved_notes = repo.get_all_notes()
    print("\n--- Reading from Database ---")
    for note in saved_notes:
        print(f"ID: {note.id}")
        print(f"Topic: {note.topic}")
        print(f"Is High Priority? {note.is_high_priority()}")
        print("-" * 20)
