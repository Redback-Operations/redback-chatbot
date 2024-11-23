import os
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain.memory.buffer import ConversationBufferMemory
from vector_db.qdrant_client import query_qdrant  # Import query_qdrant
from flashrank import RerankRequest

def init_chat_model():
    return ChatGroq(
        model_name='llama-3.1-8b-instant',
        api_key=os.getenv("GROQ_API_KEY"),
        streaming=True
    )

def generate_response(chat_model, encoder, ranker, user_input, collection_name, client):
    query_embedding = encoder.encode([user_input])[0]
    initial_results = query_qdrant(client, query_embedding, collection_name)
    rerankrequest = RerankRequest(user_input, initial_results)
    reranked_results = ranker.rerank(rerankrequest)
    context = "\n".join([result['text'] for result in reranked_results])
    TEMPLATE = """Answer the following question from the context

    context = {context}

    question = {question}
    """
    prompt_template = PromptTemplate(input_variables=["context", "question"], template=TEMPLATE)
    full_response = chat_model.predict(prompt_template.format(question=user_input, context=context))
    return full_response.strip()