from ingest.pdf_loader import convert_all_pdfs
from ingest.markdown_splitter import split_markdown_files
from ingest.embedding_generator import get_encoder, generate_embeddings
from db.qdrant_handler import init_qdrant_client, create_collection, prepare_points, upload_points

def main():
    input_folder = "docs"
    output_folder = "parsed_docs"
    collection_name = "chatbot_docs"

    convert_all_pdfs(input_folder, output_folder)
    chunks = split_markdown_files(output_folder)

    encoder = get_encoder()
    embeddings = generate_embeddings(encoder, chunks)

    client = init_qdrant_client()
    create_collection(client, collection_name)
    points = prepare_points(chunks, embeddings)
    upload_points(client, collection_name, points)

if __name__ == "__main__":
    main()
