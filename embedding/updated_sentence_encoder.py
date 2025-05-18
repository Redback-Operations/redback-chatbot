from sentence_transformers import SentenceTransformer
from typing import List, Union

def get_encoder(model_name: str = "all-MiniLM-L6-v2") -> SentenceTransformer:
    """
    Initialise and return a SentenceTransformer model.

    Args:
        model_name (str): The name of the pre-trained model to use.

    Returns:
        SentenceTransformer: Loaded embedding model.
    """
    try:
        model = SentenceTransformer(model_name)
        return model
    except Exception as e:
        raise RuntimeError(f"Failed to load model '{model_name}': {e}")

def generate_embeddings(encoder: SentenceTransformer, chunks: List[str], batch_size: int = 32) -> List[List[float]]:
    """
    Generate embeddings for a list of text chunks using the provided encoder.

    Args:
        encoder (SentenceTransformer): The embedding model to use.
        chunks (List[str]): A list of text strings to encode.
        batch_size (int): Number of texts to encode at once. Improves performance on large inputs.

    Returns:
        List[List[float]]: A list of embedding vectors.
    """
    if not chunks:
        return []

    return encoder.encode(chunks, batch_size=batch_size, show_progress_bar=True)
