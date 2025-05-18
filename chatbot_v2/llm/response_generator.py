import os
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from flashrank import RerankRequest

from db.qdrant_handler import query_collection

def init_chat_model():
    return ChatGroq(
        model_name='llama-3.1-8b-instant',
        api_key=os.getenv("GROQ_API_KEY"),
        streaming=True
    )

def generate_response(chat_model, encoder, ranker, user_input, collection_name, client, threshold=0.75):
    query_embedding = encoder.encode([user_input])[0]
    results = query_collection(client, query_embedding, collection_name)
    
    reranked = ranker.rerank(RerankRequest(user_input, results))
    filtered = [res for res in reranked if res['score'] >= threshold]
    
    context = "\n".join([res['text'] for res in filtered])
    
    if context:
        template = """Answer the following question from the context.

        Context:
        {context}

        Question:
        {question}
        """
        prompt = PromptTemplate(input_variables=["context", "question"], template=template)
        return chat_model.predict(prompt.format(context=context, question=user_input)).strip()
    else:
        prompt = PromptTemplate(input_variables=["question"], template="Answer the following question:\n{question}")
        return chat_model.predict(prompt.format(question=user_input)).strip()
