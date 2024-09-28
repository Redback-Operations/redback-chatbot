import torch
from models.language_model import BigramLanguageModel
from utils.data_loader import load_and_preprocess_data, encode_text
from training.train_model import train_model
from chatbot import chatbot_interaction


# Hyperparameters
BATCH_SIZE = 4
BLOCK_SIZE = 16
MAX_ITERS = 5000
EVAL_INTERVAL = 100
LEARNING_RATE = 1e-3
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
EVAL_ITERS = 200
N_EMBD = 64
N_HEAD = 4
N_LAYER = 4
DROPOUT = 0.0


def main():
    """
    The main function to preprocess data, train the BigramLanguageModel, and optionally
    start the chatbot interaction.

    Steps:
    1. Loads and preprocesses text data.
    2. Encodes the text into a sequence of integers.
    3. Splits the data into training and validation sets.
    4. Initializes and trains the BigramLanguageModel using transformer blocks.
    5. Saves the trained model to a file.
    6. Optionally starts an interactive chatbot session where users can ask questions
       and receive generated responses from the model.

    Hyperparameters for the model, optimizer, and training process are defined globally.

    Prompts the user to decide if they want to interact with the chatbot after training.
    """
    # Load and preprocess data
    text = load_and_preprocess_data()
    
    # Encode the text
    encoded_text, stoi, itos, vocab_size = encode_text(text)

    # Convert encoded text to a tensor and split into training and validation sets
    data = torch.tensor(encoded_text, dtype=torch.long)
    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]

    # Initialize the BigramLanguageModel
    model = BigramLanguageModel(
        vocab_size, N_EMBD, BLOCK_SIZE, N_LAYER, N_HEAD, DROPOUT
    ).to(DEVICE)

    # Initialize the optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    # Train the model
    train_model(
        model, train_data, val_data, optimizer,
        MAX_ITERS, EVAL_INTERVAL, BLOCK_SIZE, BATCH_SIZE, EVAL_ITERS
    )

    # Save the trained model
    torch.save(model.state_dict(), 'bigram_language_model.pth')

    # Prompt user to start chatbot interaction
    user_input = input("Do you want to start the chatbot? (y/n): ")
    if user_input.lower() == 'y':
        chatbot_interaction(model, stoi, itos)


if __name__ == '__main__':
    main()
