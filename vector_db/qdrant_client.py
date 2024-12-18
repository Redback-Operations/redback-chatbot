import uuid
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

def init_qdrant_client():
    """
    Initialise a Qdrant client with an in-memory database.
    """
    return QdrantClient(":memory:")

def create_collection(client, collection_name, vector_size=384):
    """
    Create a collection in Qdrant with the specified name and vector size.
    """
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
    )

def prepare_data_for_upload(chunks, embeddings):
    """
    Prepare data points for uploading to Qdrant.
    """
    ids = [str(uuid.uuid4()) for _ in range(len(chunks))]
    points = [
        PointStruct(id=id, vector=embedding, payload={"text": chunk})
        for id, embedding, chunk in zip(ids, embeddings, chunks)
    ]
    return points

def upload_data(client, collection_name, points):
    """
    Upload data points to the specified Qdrant collection.
    """
    client.upsert(collection_name=collection_name, wait=True, points=points)

def query_qdrant(client, query_embedding, collection_name, top_k=3):
    """
    Query Qdrant for the top-k nearest neighbors to the query embedding.
    """
    search_result = client.search(
        collection_name=collection_name,
        query_vector=query_embedding,
        limit=top_k,
        with_payload=True
    )
    formatted_results = [
        {"id": hit.id, "text": hit.payload.get("text", ""), "score": hit.score}
        for hit in search_result
    ]
    return formatted_results
