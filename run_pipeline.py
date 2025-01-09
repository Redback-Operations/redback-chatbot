from data_processing.pdf_to_markdown import convert_all_pdfs
from data_processing.text_splitter import split_text_files
from embedding.sentence_encoder import get_encoder, generate_embeddings
from vector_db.qdrant_client import init_qdrant_client, create_collection, prepare_data_for_upload, upload_data
from chat_model.chat_groq import init_chat_model, generate_response
from flashrank import Ranker  # Import Ranker

def main():
    """
    Main function to test the data processing pipeline.
    It converts PDFs to Markdown, splits the text into chunks,
    generates embeddings, initialises the Qdrant client, creates a collection,
    prepares data for upload, and uploads the data.
    """
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

    # Example usage
    user_query = "What is the phone number of Sport and Recreation in Alaska?"
    response = generate_response(chat_model, encoder, ranker, user_query, collection_name, client)
    print(response)

if __name__ == "__main__":
    main()
