from sentence_transformers import SentenceTransformer

def get_encoder(model_name="all-MiniLM-L6-v2"):
    """
    Initialise and return a SentenceTransformer model.
    """
    return SentenceTransformer(model_name)

def generate_embeddings(encoder, chunks):
    """
    Generate embeddings for a list of text chunks using the provided encoder.
    """
    return encoder.encode(chunks)
