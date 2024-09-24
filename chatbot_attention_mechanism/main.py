import torch
from models.language_model import BigramLanguageModel
from utils.data_loader import load_and_preprocess_data, encode_text
from training.train_model import train_model
from chatbot import chatbot_interaction


# Hyperparameters
batch_size = 4
block_size = 16
max_iters = 5000
eval_interval = 100
learning_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 64
n_head = 4
n_layer = 4
dropout = 0.0


def main():
    text = load_and_preprocess_data()
    encoded_text, stoi, itos, vocab_size = encode_text(text)

    data = torch.tensor(encoded_text, dtype=torch.long)
    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]

    model = BigramLanguageModel(
        vocab_size, n_embd, block_size, n_layer, n_head, dropout
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    train_model(
        model, train_data, val_data, optimizer,
        max_iters, eval_interval, block_size, batch_size, eval_iters
    )

    torch.save(model.state_dict(), 'bigram_language_model.pth')

    user_input = input("Do you want to start the chatbot? (y/n): ")
    if user_input.lower() == 'y':
        chatbot_interaction(model, stoi, itos)


if __name__ == '__main__':
    main()
