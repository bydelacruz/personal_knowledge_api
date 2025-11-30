"""
data model for note entries
"""

from dataclasses import dataclass


@dataclass
class NoteEntry:
    """
    Represents a note entry with topic, tags, and rating.
    """

    topic: str
    tags: list[str]
    rating: int
    id: int | None = None

    def is_high_priority(self) -> bool:
        """
        determin if note is high priority based on rating
        """
        return self.rating >= 8
