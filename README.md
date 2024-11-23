# Redback Chatbot Development Project

Welcome to the **Redback Chatbot Development Project**, part of the larger **Redback Senior Project**. This project is dedicated to harnessing the potential of wearable technology to significantly enhance the quality of life for the elderly. By leveraging advanced data analytics, innovative web platforms, and sophisticated mobile app development tools, the Redback Senior Project aims to create a comprehensive support system for seniors.

This repository specifically focuses on the development of an AI-powered chatbot that complements the wearable technology, providing personalised interactions across domains such as mental health, medical information, and natural language processing (NLP). The chatbot is split into two major parts:

1. **Model Development** – The core machine learning model, including natural language processing techniques and training datasets.
2. **RAG Implementation** – Retrieval-Augmented Generation (RAG) for real-time data retrieval and contextual response generation from a curated knowledge base.

## Introduction

The goal of the Redback Chatbot is to develop an intelligent, adaptive chatbot that goes beyond traditional transformer models. By integrating **LLMs** and **RAG** architecture, the chatbot retrieves and generates responses from a specialised knowledge base to assist elderly users, enhancing their interactions with the Redback wearable technology.

## Features

- **Advanced Tokenisation:** Utilises subword tokenisation (BPE, WordPiece) for more accurate text representation.
- **Vector Embedding:** Leverages Word2Vec, GloVe, and fastText to generate meaningful semantic embeddings.
- **Attention Mechanism:** Ensures focused, high-quality responses by attending to key parts of the input data.
- **Iterative Refinement:** Constantly refines and optimises performance to meet the evolving needs of the elderly.
- **RAG Architecture:** Combines document retrieval with generation models for contextually relevant, real-time responses.
- **LLM Integration:** Allows for dynamic, personalised, and context-aware conversations.

## Project Structure

The repository is divided into two parts, reflecting the two main components of the chatbot:

1. **Model Development:**
   - Core implementation of the chatbot’s machine learning model.
   - Includes the training and integration of large language models.
   - **Chatbot_datasets/**: Datasets used for training the model, including:
     - `Medical_train.csv`, `Mental_Health_FAQ.csv`, etc.
   - **Chatbot_model/**: Main implementation of the chatbot model.
     - `chat.py`

2. **RAG Implementation:**
   - Focuses on implementing Retrieval-Augmented Generation to enhance the chatbot’s contextual understanding.
   - **rag/**: Implementation of the RAG architecture

Other key files include:
- **CODEOWNERS**: Designates code ownership.
- **LICENSE**: Licensing information.

## Setup Instructions

To set up the project locally, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/redback-chatbot.git
   ```
### Part 1: Model Development

(Additional usage instructions to be added)

### Part 2: RAG Implementation

2. **Run Jupyter Notebook:**
   Launch Jupyter Notebook in the `rag/` directory:
   ```bash
   jupyter notebook
   ```
   Open `poc_e2e.ipynb` from the Jupyter interface.

3. **Follow Instructions in the Notebook:**
   The notebook contains all the instructions needed to run the end-to-end process including environment set-up and installing dependencies provided in `rag/requirements.txt`.

## Usage

### Part 1: Model Development

(Additional usage instructions to be added)

### Part 2: RAG Implementation

RAG architecture is employed to enhance the chatbot’s ability to retrieve and generate contextually accurate responses. 

1. **Convert PDF Files to Markdown**: Place your PDF files in the `docs` folder. The notebook will automatically convert them to Markdown format and save them in the `parsed_docs` folder.
2. **Text Processing**: The notebook splits the converted Markdown files into smaller text chunks, which are then encoded into vector embeddings using the `SentenceTransformer` model.
3. **Store Embeddings in Qdrant**: The text embeddings are stored in a Qdrant vector database, enabling efficient semantic search.
4. **Search and Retrieve**: Use the stored embeddings to perform searches. The results are initially ranked based on relevance and can be further refined using a ranking model.
5. **Generate Responses**: Use the `ChatGroq` model to generate answers based on the retrieved context and user queries.

---

This chatbot is part of the overall Redback Senior initiative, aimed at creating intelligent support systems for the elderly, ensuring their well-being, and providing them with reliable access to important information through wearable technology.

Feel free to explore, contribute, and be part of this important initiative!