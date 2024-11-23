from sentence_transformers import SentenceTransformer

def get_encoder(model_name="all-MiniLM-L6-v2"):
    return SentenceTransformer(model_name)

def generate_embeddings(encoder, chunks):
    return encoder.encode(chunks)