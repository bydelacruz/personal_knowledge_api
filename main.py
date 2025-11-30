"""
main entry point for notes application
"""

from dataclasses import dataclass


@dataclass
class KnowledgeEntry:
    """
    Represents a knowledge entry with topic, tags, and rating.
    """

    topic: str
    tags: list[str]
    rating: int

    def is_high_priority(self) -> bool:
        """
        Determines if the knowledge entry is high priority based on its rating.
        """
        return self.rating >= 8


if __name__ == "__main__":
    note = KnowledgeEntry("python", ["learning", "backend"], 9)

    print(f"Is this note high priority? {note.is_high_priority()}")
