from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import logging

from ingest.markdown_splitter import split_markdown_files
from ingest.embedding_generator import get_encoder, generate_embeddings
from db.qdrant_handler import init_qdrant_client, create_collection, prepare_points, upload_points
from llm.response_generator import init_chat_model, generate_response
from flashrank import Ranker

logging.basicConfig(level=logging.INFO)
app = FastAPI()

COLLECTION_NAME = "chatbot_docs"
encoder = get_encoder()
client = init_qdrant_client()
chat_model = init_chat_model()
ranker = Ranker()

class Query(BaseModel):
    user_query: str

@app.post("/chat")
async def chat(query: Query):
    try:
        response = generate_response(chat_model, encoder, ranker, query.user_query, COLLECTION_NAME, client)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
