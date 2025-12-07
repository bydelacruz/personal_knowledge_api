from ingestion import chunk_text, extract_text_from_pdf

pdf_path = "/Users/benjamindelacruz/Documents/Resume_Benjamin_Delacruz.pdf"

full_text = extract_text_from_pdf(pdf_path)

chunks = chunk_text(full_text)

print(f"total length of text is: {len(full_text)}")
print(f"number of chunks: {len(chunks)}")
print(f"first chunk content ---- \n {chunks[0]}")
