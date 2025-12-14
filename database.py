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
            # Table 1: Notes
            await db.execute("""
                CREATE TABLE IF NOT EXISTS notes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                topic TEXT,
                tags TEXT,
                rating INTEGER
                )
            """)

            # Table 2: Users
            # We store the username and the 'hashed' password
            await db.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL
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

    async def create_user(self, username, password_hash):
        try:
            async with aiosqlite.connect(self.db_name) as db:
                await db.execute(
                    "INSERT INTO users (username, password_hash) VALUES (?,?)",
                    (username, password_hash),
                )
                await db.commit()
            return True
        except aiosqlite.IntegrityError:
            # This happens if the username already exists (UNIQUE constraint)
            return False

    async def get_user_by_username(self, username):
        async with aiosqlite.connect(self.db_name) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                "SELECT * FROM users WHERE username = ?", (username,)
            )
            return await cursor.fetchone()

    def _row_to_entry(self, row) -> NoteEntry:
        """
        Converts a database row to a NoteEntry object.
        """
        row_id, topic, tags_string, rating = row
        tags_list = tags_string.split(",") if tags_string else []

        return NoteEntry(id=row_id, topic=topic, tags=tags_list, rating=rating)
