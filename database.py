"""
A simple note-taking repository using SQLite.
"""

import sqlite3


class NoteRepository:
    """
    A repository for managing notes in an SQLite database.
    Each note has a topic and a rating.
    """

    def __init__(self, filename: str = "notes.db"):
        self.db_name = filename
        self.create_table()

    def connect(self):
        """
        Establishes a connection to the SQLite database.
        """
        return sqlite3.connect(self.db_name)

    def create_table(self):
        """
        Creates the notes table if it does not exist.
        """
        with self.connect() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS notes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                topic TEXT,
                rating INTEGER
                )
            """)
            conn.commit()

    def add_note(self, topic: str, rating: int):
        """
        Adds a new note to the database.
        """
        with self.connect() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO notes (topic, rating) VALUES(?,?)", (topic, rating)
            )
            conn.commit()

    def get_all_notes(self) -> list[tuple]:
        """
        Retrieves all notes from the database.
        """
        with self.connect() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM notes")
            notes = cursor.fetchall()

            return notes


if __name__ == "__main__":
    # 1. Initialize the DB
    repo = NoteRepository()

    # 2. Add some data
    print("Adding notes...")
    repo.add_note("Python Generators", 9)
    repo.add_note("SQL Injections", 10)
    repo.add_note("Bad Topic", 2)

    # 3. Read it back
    print("Fetching notes...")
    all_notes = repo.get_all_notes()

    for note in all_notes:
        print(note)
