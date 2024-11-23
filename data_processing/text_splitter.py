import os
from langchain.text_splitter import MarkdownTextSplitter

def split_text_files(docs_path, chunk_size=500, chunk_overlap=50):
    text_splitter = MarkdownTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    documents = [
        os.path.join(docs_path, file)
        for file in os.listdir(docs_path)
        if file.endswith('.md')
    ]
    chunks = []
    for doc in documents:
        with open(doc, 'r', encoding='utf-8') as file:
            text = file.read()
            chunk = text_splitter.split_text(text)
            chunks.extend(chunk)
    return chunks