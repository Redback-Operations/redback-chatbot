from sentence_transformers import SentenceTransformer

def get_encoder():
    return SentenceTransformer('all-MiniLM-L6-v2')

def generate_embeddings(encoder, texts: list) -> list:
    return encoder.encode(texts, show_progress_bar=True)
