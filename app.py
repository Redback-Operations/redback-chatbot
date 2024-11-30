import os
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from data_processing.pdf_to_markdown import convert_all_pdfs
from data_processing.text_splitter import split_text_files
from embedding.sentence_encoder import get_encoder, generate_embeddings
from vector_db.qdrant_client import init_qdrant_client, create_collection, prepare_data_for_upload, upload_data, query_qdrant
from chat_model.chat_groq import init_chat_model, generate_response
from flashrank import Ranker


# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Set up paths
input_folder = 'docs'
output_folder = 'parsed_docs'
collection_name = 'my_text_chunks'

# Convert PDFs to Markdown
convert_all_pdfs(input_folder, output_folder)

# Split Markdown files into chunks
chunks = split_text_files(output_folder)

# Generate embeddings
encoder = get_encoder()
embeddings = generate_embeddings(encoder, chunks)

# Initialise Qdrant client and create collection
client = init_qdrant_client()
create_collection(client, collection_name)

# Prepare and upload data
points = prepare_data_for_upload(chunks, embeddings)
upload_data(client, collection_name, points)

# Initialise ChatGroq model
chat_model = init_chat_model()

# Initialise Ranker
ranker = Ranker()

class Query(BaseModel):
    user_query: str

@app.post("/chat")
async def chat(query: Query):
    try:
        logger.info(f"Received query: {query.user_query}")
        response = generate_response(chat_model, encoder, ranker, query.user_query, collection_name, client)
        logger.info(f"Generated response: {response}")
        return {"response": response}
    except Exception as e:
        logger.error(f"Error generating response: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)