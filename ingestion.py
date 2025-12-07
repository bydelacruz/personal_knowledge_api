import pypdf


def extract_text_from_pdf(file_path: str) -> str:
    """
    Opens a PDF file and mashes all pages into one giant string.
    """
    reader = pypdf.PdfReader(file_path)
    full_text = ""
    for page in reader.pages:
        full_text += page.extract_text() + "\n"
    return full_text


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """
    Splits text into smaller pieces.

    Args:
        text: The massive string from the PDF.
        chunk_size: How many characters per chunk.
        overlap: How many characters to repeat (to prevent cutting sentences awkwardly).

    Returns:
        A list of string chunks.
    """
    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        # Calculate the end point
        end = start + chunk_size

        # Slice the text
        chunk = text[start:end]

        # Save it
        chunks.append(chunk)

        # Move the window forward, but step back by the overlap amount
        # This creates the "sliding window" effect
        start += chunk_size - overlap

    return chunks
