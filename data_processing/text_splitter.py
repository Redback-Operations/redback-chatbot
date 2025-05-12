from pathlib import Path
from langchain.text_splitter import MarkdownTextSplitter

def split_text_files(docs_path, chunk_size=500, chunk_overlap=50):
    """
    Splits all Markdown (.md) files in a directory into smaller text chunks using LangChain's MarkdownTextSplitter.

    Args:
        docs_path (str or Path): Path to the directory containing markdown files.
        chunk_size (int): Maximum number of characters in each chunk.
        chunk_overlap (int): Number of overlapping characters between chunks.

    Returns:
        List[str]: A list of text chunks extracted from all markdown files.
    """
    docs_path = Path(docs_path)
    text_splitter = MarkdownTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = []

    for file_path in docs_path.glob("*.md"):
        try:
            print(f"Processing file: {file_path.name}")
            text = file_path.read_text(encoding='utf-8')
            chunked_text = text_splitter.split_text(text)
            chunks.extend(chunked_text)
        except Exception as e:
            print(f"Error processing {file_path.name}: {e}")

    return chunks
