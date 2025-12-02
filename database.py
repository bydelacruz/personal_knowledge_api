"""
A simple note-taking repository using SQLite.
"""

import aiosqlite

from models import NoteEntry


class NoteRepository:
    """
    A repository for managing notes in an SQLite database.
    Each note has a topic and a rating.
    """

    def __init__(self, filename: str = "notes.db"):
        self.db_name = filename

    async def create_table(self):
        """
        Creates the notes table if it does not exist.
        """
        async with aiosqlite.connect(self.db_name) as db:
            await db.execute("""
                CREATE TABLE IF NOT EXISTS notes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                topic TEXT,
                tags TEXT,
                rating INTEGER
                )
            """)
            await db.commit()

    async def add_note(self, note: NoteEntry) -> int:
        """
        Adds a new note to the database.
        """
        async with aiosqlite.connect(self.db_name) as db:
            tags_string = ",".join(note.tags)
            cursor = await db.execute(
                "INSERT INTO notes (topic, tags, rating) VALUES(?,?,?)",
                (note.topic, tags_string, note.rating),
            )
            await db.commit()
            return cursor.lastrowid

    async def get_all_notes(self) -> list[NoteEntry]:
        """
        Retrieves all notes from the database.
        """
        async with aiosqlite.connect(self.db_name) as db:
            async with db.execute("SELECT * FROM notes") as cursor:
                rows = await cursor.fetchall()

            results = []
            for row in rows:
                results.append(self._row_to_entry(row))

            return results

    def _row_to_entry(self, row) -> NoteEntry:
        """
        Converts a database row to a NoteEntry object.
        """
        row_id, topic, tags_string, rating = row
        tags_list = tags_string.split(",") if tags_string else []

        return NoteEntry(id=row_id, topic=topic, tags=tags_list, rating=rating)
