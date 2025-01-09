import torch
from utils.data_loader import encode_text

def chatbot_interaction(model, stoi, itos):
    """
    This function facilitates an interactive chatbot session.

    Args:
    - model: A pre-trained language model used to generate responses.
    - stoi: A dictionary that maps characters to their respective indices.
    - itos: A dictionary that maps indices back to their respective characters.

    The function repeatedly prompts the user to input a question, processes it using the model,
    and then generates a response based on the input. The interaction continues until the user 
    types 'quit' to exit. It encodes the user's question using `stoi`, generates a response from 
    the model, and then decodes the generated response back into text using `itos`.

    The model generates a response with a maximum of 1000 tokens and a block size of 16. 
    """
    while True:
        question = input("Enter your question (or type 'quit' to exit): ")
        if question.lower() == 'quit':
            break

        encoded_question, _, _, _ = encode_text(question)
        encoded_question = torch.tensor(
            encoded_question, dtype=torch.long
        ).unsqueeze(0)

        generated_response = model.generate(
            encoded_question, max_new_tokens=1000, block_size=16
        )[0].tolist()

        decoded_response = ''.join([itos[i] for i in generated_response])
        print("Generated response:", decoded_response)
