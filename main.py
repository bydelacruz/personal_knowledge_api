from dataclasses import dataclass


@dataclass
class KnowledgeEntry:
    topic: str
    tags: list[str]
    rating: int

    def is_high_priority(self) -> bool:
        if self.rating >= 8:
            return True
        return False


if __name__ == "__main__":
    note = KnowledgeEntry("python", ["learning", "backend"], 9)

    print(f"Is this note high priority? {note.is_high_priority()}")
