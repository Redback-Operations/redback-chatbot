import uuid
from typing import List, Dict, Any
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

def init_qdrant_client() -> QdrantClient:
    """
    Initialise a Qdrant client with an in-memory database.
    
    Returns:
        QdrantClient: A client connected to an in-memory Qdrant instance.
    """
    return QdrantClient(":memory:")

def create_collection(client: QdrantClient, collection_name: str, vector_size: int = 384) -> None:
    """
    Create a collection in Qdrant.

    Args:
        client (QdrantClient): The Qdrant client.
        collection_name (str): The name of the collection.
        vector_size (int): The size of the vectors to be stored.
    """
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
    )

def prepare_data_for_upload(chunks: List[str], embeddings: List[List[float]]) -> List[PointStruct]:
    """
    Prepare PointStruct objects for uploading to Qdrant.

    Args:
        chunks (List[str]): The list of text chunks.
        embeddings (List[List[float]]): The list of vector embeddings.

    Returns:
        List[PointStruct]: A list of points ready for upload.
    """
    ids = [str(uuid.uuid4()) for _ in range(len(chunks))]
    return [
        PointStruct(id=id, vector=embedding, payload={"text": chunk})
        for id, embedding, chunk in zip(ids, embeddings, chunks)
    ]

def upload_data(client: QdrantClient, collection_name: str, points: List[PointStruct]) -> None:
    """
    Upload points to a Qdrant collection.

    Args:
        client (QdrantClient): The Qdrant client.
        collection_name (str): Name of the collection.
        points (List[PointStruct]): Points to upload.
    """
    client.upsert(collection_name=collection_name, wait=True, points=points)

def query_qdrant(client: QdrantClient, query_embedding: List[float], collection_name: str, top_k: int = 3) -> List[Dict[str, Any]]:
    """
    Query Qdrant to retrieve top-k similar vectors.

    Args:
        client (QdrantClient): The Qdrant client.
        query_embedding (List[float]): The embedding to query against.
        collection_name (str): The name of the collection to query.
        top_k (int): Number of results to return.

    Returns:
        List[Dict[str, Any]]: List of matched results with id, text, and similarity score.
    """
    search_result = client.search(
        collection_name=collection_name,
        query_vector=query_embedding,
        limit=top_k,
        with_payload=True
    )
    return [
        {
            "id": hit.id,
            "text": hit.payload.get("text", ""),
            "score": hit.score
        }
        for hit in search_result
    ]
