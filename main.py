"""
main entry point for notes application
"""

from dataclasses import dataclass


@dataclass
class NoteEntry:
    """
    Represents a knowledge entry with topic, tags, and rating.
    """

    topic: str
    tags: list[str]
    rating: int

    def is_high_priority(self) -> bool:
        """
        determing if note is high priority based on rating
        """
        return self.rating >= 8


if __name__ == "__main__":
    note = NoteEntry("python", ["learning", "backend"], 9)

    print(f"Is this note high priority? {note.is_high_priority()}")
