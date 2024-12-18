import os
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from vector_db.qdrant_client import query_qdrant  # Import query_qdrant
from flashrank import RerankRequest

def init_chat_model():
    """
    Initialise the ChatGroq model with the specified parameters.
    """
    return ChatGroq(
        model_name='llama-3.1-8b-instant',
        api_key=os.getenv("GROQ_API_KEY"),
        streaming=True
    )

def generate_response(chat_model, encoder, ranker, user_input, collection_name, client, relevance_threshold=0.75):
    """
    Generate a response based on user input by querying and reranking results from a vector database.
    """
    
    # Encode the user input
    query_embedding = encoder.encode([user_input])[0]
    
    # Query the Qdrant vector database
    initial_results = query_qdrant(client, query_embedding, collection_name)
    
    # Rerank the initial results
    rerankrequest = RerankRequest(user_input, initial_results)
    reranked_results = ranker.rerank(rerankrequest)
    
    # Filter results based on relevance threshold
    filtered_results = [result for result in reranked_results if result['score'] >= relevance_threshold]
    
    # Debugging for filtering
    print(f"Filtered: {filtered_results}")

    # Construct the context from filtered results
    context = "\n".join([result['text'] for result in filtered_results])
    
    # Define the prompt template
    if context:
        template = """Answer the following question from the context

        context = {context}

        question = {question}
        """
        prompt_template = PromptTemplate(input_variables=["context", "question"], template=template)
        full_response = chat_model.predict(prompt_template.format(question=user_input, context=context))
    else:
        TEMPLATE = """Answer the following question

        question = {question}
        """
        prompt_template = PromptTemplate(input_variables=["question"], template=template)
        full_response = chat_model.predict(prompt_template.format(question=user_input))
    
    return full_response.strip()
