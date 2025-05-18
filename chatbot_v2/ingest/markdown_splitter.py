import os
from langchain.text_splitter import MarkdownTextSplitter

def split_markdown_files(folder_path: str, chunk_size: int = 500, chunk_overlap: int = 50) -> list:
    splitter = MarkdownTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.md'):
            with open(os.path.join(folder_path, filename), 'r', encoding='utf-8') as file:
                text = file.read()
                chunks.extend(splitter.split_text(text))
    return chunks
