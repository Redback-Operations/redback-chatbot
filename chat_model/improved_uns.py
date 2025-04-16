import os
from typing import List

from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from vector_db.qdrant_client import query_qdrant
from flashrank import RerankRequest

# --- Prompt templates ---
ANSWER_WITH_CONTEXT_TEMPLATE = """Answer the following question using the context provided.

Context:
{context}

Question:
{question}
"""

ANSWER_WITHOUT_CONTEXT_TEMPLATE = """Answer the following question.

Question:
{question}
"""

# --- Initialize chat model ---
def init_chat_model() -> ChatGroq:
    """
    Initialise the ChatGroq model with API key and model name.
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY environment variable is not set.")

    return ChatGroq(
        model_name='llama-3.1-8b-instant',
        api_key=api_key,
        streaming=False  # Set to True only if using stream consumption
    )

# --- Filter results by score ---
def filter_results(results: List[dict], threshold: float) -> List[dict]:
    return [r for r in results if r['score'] >= threshold]

# --- Construct context string ---
def build_context(results: List[dict]) -> str:
    return "\n".join([result['text'] for result in results])

# --- Build appropriate prompt ---
def build_prompt(context: str, user_input: str) -> PromptTemplate:
    if context:
        prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template=ANSWER_WITH_CONTEXT_TEMPLATE
        )
        return prompt_template.format_prompt(question=user_input, context=context)
    else:
        prompt_template = PromptTemplate(
            input_variables=["question"],
            template=ANSWER_WITHOUT_CONTEXT_TEMPLATE
        )
        return prompt_template.format_prompt(question=user_input)

# --- Main response function ---
def generate_response(
    chat_model: ChatGroq,
    encoder,
    ranker,
    user_input: str,
    collection_name: str,
    client,
    relevance_threshold: float = 0.75
) -> str:
    """
    Generate a response to the user query using vector search and reranking.
    """

    try:
        # Step 1: Encode the user query
        query_embedding = encoder.encode([user_input])[0]
    except Exception as e:
        print(f"[Encoding Error] {e}")
        return "Sorry, there was a problem processing your question."

    try:
        # Step 2: Query the vector database
        initial_results = query_qdrant(client, query_embedding, collection_name)
    except Exception as e:
        print(f"[Qdrant Query Error] {e}")
        return "Sorry, I couldn't retrieve relevant information from the database."

    try:
        # Step 3: Rerank results
        rerank_request = RerankRequest(user_input, initial_results)
        reranked_results = ranker.rerank(rerank_request)
    except Exception as e:
        print(f"[Reranking Error] {e}")
        return "Sorry, I encountered an error while prioritizing the results."

    # Step 4: Filter by relevance
    filtered_results = filter_results(reranked_results, relevance_threshold)
    print(f"[INFO] Initial: {len(initial_results)} | Filtered: {len(filtered_results)}")

    if not filtered_results:
        return "I couldn’t find anything highly relevant. Could you try rephrasing your question?"

    # Step 5: Build context and prompt
    context = build_context(filtered_results)
    prompt = build_prompt(context, user_input)

    # Step 6: Generate response using the chat model
    try:
        return chat_model.predict(prompt.to_string()).strip()
    except Exception as e:
        print(f"[Chat Model Error] {e}")
        return "Something went wrong while generating a response. Please try again."

