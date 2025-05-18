from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
import uuid

def init_qdrant_client():
    return QdrantClient(host="localhost", port=6333)

def create_collection(client, collection_name: str, vector_size: int = 384):
    client.recreate_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
    )

def prepare_points(text_chunks: list, embeddings: list) -> list:
    return [
        PointStruct(id=str(uuid.uuid4()), vector=embedding, payload={"text": chunk})
        for chunk, embedding in zip(text_chunks, embeddings)
    ]

def upload_points(client, collection_name: str, points: list):
    client.upsert(collection_name=collection_name, points=points)

def query_collection(client, embedding: list, collection_name: str, top_k: int = 5) -> list:
    hits = client.search(collection_name=collection_name, query_vector=embedding, limit=top_k)
    return [{"text": hit.payload["text"], "score": hit.score} for hit in hits]
