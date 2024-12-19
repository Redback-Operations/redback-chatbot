# Lachesis Chatbot Development Project

Welcome to the **Lachesis Chatbot Development Project**, part of the larger **Redback Senior Project (Lachesis)** under Redback Operations (Capstone project company within the School of Information Technology at Deakin University). This project is dedicated to harnessing the potential of wearable technology to significantly enhance the quality of life for the elderly. By leveraging advanced data analytics, innovative web platforms, and sophisticated mobile app development tools, the Redback Senior Project aims to create a comprehensive support system for seniors.

This repository specifically focuses on the development of an AI-powered chatbot that complements the wearable technology, providing personalised interactions across medical and personal health domains.

## Introduction

The Lachesis Chatbot is designed to assist elderly users by providing them with reliable information and support through natural language interactions. The chatbot leverages advanced machine learning models and a retrieval-augmented generation (RAG) architecture to deliver accurate and contextually relevant responses.

## Features

- **LLM Integration**: Integrates large language models (LLMs) for dynamic, personalised, and context-aware conversations.
- **Vector Embedding**: Utilises `SentenceTransformer` for generating meaningful semantic embeddings, ensuring accurate text representation.
- **RAG Architecture**: Combines document retrieval with generation models for contextually relevant, real-time responses.
- **Data Processing Pipeline**: Includes scripts for converting PDFs to Markdown, splitting text, generating embeddings, and uploading data to Qdrant.
- **User-Friendly Interface**: Provides an easy-to-use interface for users through a Streamlit-based frontend.

## Project Structure

```
project/
│
├── data_processing/
│   ├── __init__.py
│   ├── pdf_to_markdown.py
│   ├── text_splitter.py
│   └── utils.py
│
├── embedding/
│   ├── __init__.py
│   ├── sentence_encoder.py
│
├── vector_db/
│   ├── __init__.py
│   ├── qdrant_client.py
│
├── chat_model/
│   ├── __init__.py
│   ├── chat_groq.py
│
├── run_pipeline.py      # Script to test the processing pipeline          
├── chatbot_ui.py        # Streamlit application for the chatbot UI
├── chatbot_api.py       # FastAPI server for the chatbot
└── requirements.txt     # Project dependencies
```

## Setup Instructions

### Prerequisites

- Docker
- Visual Studio Code (VS Code) with the Dev Containers extension

### Step-by-Step Setup Using Docker

1. Install Docker Desktop
2. Enable WSL 2
3. Start Docker Desktop
4. Verify Docker Installation
6. Build the Docker Image
7. Run the Docker Container locally

### Step-by-Step Setup Using Dev Containers in VS Code

1. Install Visual Studio Code
2. Install the Dev Containers Extension
3. Clone the Repository
4. Reopen in Container
   - VS Code will build the Docker container and open the project inside the container.

### Environment Variables

Set the necessary environment variables for the Groq API key:
```sh
export GROQ_API_KEY=your_groq_api_key
```

### Test the processing pipeline in the terminal
```
python run_pipeline.py
```

## Usage

1. Running the FastAPI Server:
   - Convert PDFs to Markdown, split text, generate embeddings, and upload data to Qdrant
   - Start the FastAPI server to handle API requests
   ```
   python chatbot_api.py
   ```

2. Running the Streamlit Application:
   - Using a different terminal, start the Streamlit application to provide a user interface for the chatbot
   ```
   streamlit run chatbot_ui.py --server.address 127.0.0.1
   ```

3. Interact with the Chatbot:
   - Open your web browser and go to http://localhost:8501
   - Enter your query in the input box and click "Get Response" to receive a response from the chatbot.

---

This chatbot is part of the overall Redback Senior initiative, aimed at creating intelligent support systems for the elderly, ensuring their well-being, and providing them with reliable access to important information through wearable technology.

Feel free to explore, contribute, and be part of this important initiative!