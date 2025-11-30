"""
A simple note-taking repository using SQLite.
"""

import sqlite3

from models import NoteEntry


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
                tags TEXT,
                rating INTEGER
                )
            """)
            conn.commit()

    def add_note(self, note: NoteEntry) -> int:
        """
        Adds a new note to the database.
        """
        with self.connect() as conn:
            cursor = conn.cursor()
            tags_string = ",".join(note.tags)
            cursor.execute(
                "INSERT INTO notes (topic, tags, rating) VALUES(?,?,?)",
                (note.topic, tags_string, note.rating),
            )
            conn.commit()

            return cursor.lastrowid

    def get_all_notes(self) -> list[NoteEntry]:
        """
        Retrieves all notes from the database.
        """
        with self.connect() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM notes")
            rows = cursor.fetchall()

            results = []
            for row in rows:
                new_entry = self._row_to_entry(row)
                results.append(new_entry)

            return results

    def _row_to_entry(self, row) -> NoteEntry:
        """
        Converts a database row to a NoteEntry object.
        """
        row_id, topic, tags_string, rating = row

        tags_list = tags_string.split(",") if tags_string else []

        return NoteEntry(id=row_id, topic=topic, tags=tags_list, rating=rating)
