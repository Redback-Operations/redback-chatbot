# redback-chatbot

# Chatbot Development Project

Welcome to the Chatbot Development Project!

This repository contains the implementation of a chatbot designed to answer questions related to various topics, including mental health, medical information, and natural language processing (NLP). The chatbot utilizes machine learning and deep learning techniques to understand and respond to user queries effectively.

## Introduction

This project aims to build an intelligent chatbot capable of understanding and responding to user queries with high accuracy. The chatbot leverages advanced tokenization and vector embedding techniques, as well as attention mechanisms, to improve its conversational capabilities.

## Features

- **Advanced Tokenization**: Utilizes subword tokenization methods like Byte Pair Encoding (BPE) and WordPiece.
- **Vector Embedding**: Implements word embedding techniques such as Word2Vec, GloVe, and fastText.
- **Attention Mechanism**: Enhances the chatbot's ability to focus on relevant parts of the input data.
- **Encoder-Decoder Architecture**: Uses sophisticated models for better context understanding and response generation.
- **Iterative Improvement**: Continuously evaluates and refines the model for optimal performance.

Repository Structure
.ipynb_checkpoints/: Contains checkpoint files for the Jupyter notebook.
Chatbot_datasets/: Directory containing various datasets used for training the chatbot.
Medical_train.csv
Mental_Health_FAQ.csv
NLP_mental_health_Testing.csv
concatenated_data.csv
full_Chat_data.csv
medquad.csv
Chatbot_model/: Directory containing the main chatbot implementation.
chat.py
CODEOWNERS: Defines the code owners for the repository.
LICENSE: License file for the project.

To set up the project locally, follow these steps:

1. **Clone the repository**:
   git clone https://github.com/priyabsah/redback-chatbot.git
cd redback-chatbot


2. **Install dependencies**:
    pip install -r requirements.txt


. **Download and prepare datasets**:
    - Ensure you have the necessary datasets in the `Chatbot-dataset` directory.
    - Preprocess the datasets by running:
      ```bash
      python preprocess_data.py
      ```

## Usage

Training the Chatbot
The chatbot is trained using various datasets containing questions and answers related to different domains. The training process involves the following steps:

Data Preprocessing:

Remove duplicates, handle missing values, and standardize data formats.
Tokenize and vectorize the text data using techniques like TF-IDF.
Model Training:

Implement a baseline model using LinearSVC.
Explore and integrate advanced algorithms and techniques, such as deep learning models and attention mechanisms.
Evaluation and Iteration:

Evaluate the model's performance using metrics like precision, recall, and F1-score.
Iterate and refine the model based on evaluation results.
Running the Chatbot
To run the chatbot, execute the following command:
python Chatbot_model/chat.py
