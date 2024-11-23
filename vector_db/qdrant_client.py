import uuid
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

def init_qdrant_client():
    return QdrantClient(":memory:")

def create_collection(client, collection_name, vector_size=384):
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
    )

def prepare_data_for_upload(chunks, embeddings):
    ids = [str(uuid.uuid4()) for _ in range(len(chunks))]
    points = [
        PointStruct(id=id, vector=embedding, payload={"text": chunk})
        for id, embedding, chunk in zip(ids, embeddings, chunks)
    ]
    return points

def upload_data(client, collection_name, points):
    client.upsert(collection_name=collection_name, wait=True, points=points)

def query_qdrant(client, query_embedding, collection_name, top_k=3):
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