import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from data_processing.pdf_to_markdown import convert_all_pdfs
from data_processing.text_splitter import split_text_files
from embedding.sentence_encoder import get_encoder, generate_embeddings
from vector_db.qdrant_client import init_qdrant_client, create_collection, prepare_data_for_upload, upload_data
from chat_model.chat_groq import init_chat_model, generate_response
from flashrank import Ranker


# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Set up paths
INPUT_FOLDER = 'docs'
OUTPUT_FOLDER = 'parsed_docs'
COLLECTION_NAME = 'my_text_chunks'

# Convert PDFs to Markdown
convert_all_pdfs(INPUT_FOLDER, OUTPUT_FOLDER)

# Split Markdown files into chunks
chunks = split_text_files(OUTPUT_FOLDER)

# Generate embeddings
encoder = get_encoder()
embeddings = generate_embeddings(encoder, chunks)

# Initialise Qdrant client and create collection
client = init_qdrant_client()
create_collection(client, COLLECTION_NAME)

# Prepare and upload data
points = prepare_data_for_upload(chunks, embeddings)
upload_data(client, COLLECTION_NAME, points)

# Initialise ChatGroq model
chat_model = init_chat_model()

# Initialise Ranker
ranker = Ranker()

class Query(BaseModel):
    """
    Query model for user input.
    """
    user_query: str

@app.post("/chat")
async def chat(query: Query):
    """
    Endpoint to handle chat queries.
    """
    try:
        logger.info("Received query: %s", query.user_query)
        response = generate_response(chat_model, encoder, ranker, query.user_query, COLLECTION_NAME, client)
        logger.info("Generated response: %s", response)
        return {"response": response}
    except Exception as e:
        logger.error("Error generating response: %s", str(e))
        raise HTTPException(status_code=500, detail=str(e)) from e

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
