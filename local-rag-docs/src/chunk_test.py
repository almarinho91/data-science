from src.utils import chunk_text

txt = "a" * 307
chunks = chunk_text(txt, chunk_size=500, overlap=80)
print(len(chunks))
print([len(c.text) for c in chunks])