import torch
#from models.language_model import BigramLanguageModel
from utils.data_loader import encode


def chatbot_interaction(model, stoi, itos):
    while True:
        question = input("Enter your question (or type 'quit' to exit): ")
        if question.lower() == 'quit':
            break

        encoded_question = torch.tensor(
            encode(question, stoi), dtype=torch.long
        ).unsqueeze(0)

        generated_response = model.generate(
            encoded_question, max_new_tokens=1000, block_size=16
        )[0].tolist()

        decoded_response = ''.join([itos[i] for i in generated_response])
        print("Generated response:", decoded_response)
