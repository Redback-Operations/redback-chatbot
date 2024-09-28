import torch
from torch import nn  # Refactored as suggested
from torch.nn import functional as F


class Head(nn.Module):
    """
    A single attention head that performs self-attention on the input sequence.

    Args:
    - head_size (int): The dimension of the key, query, and value vectors for this head.
    - n_embd (int): The size of the input embedding.
    - block_size (int): The length of the sequence to consider (maximum number of tokens).
    - dropout (float): Dropout rate to use after applying the attention weights.

    Methods:
    - forward(x): Applies self-attention to the input `x` and returns the result.
    """
    def __init__(self, head_size, n_embd, block_size, dropout):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer(
            'tril', torch.tril(torch.ones(block_size, block_size))
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Performs forward pass of self-attention for a single head.

        Args:
        - x (Tensor): Input tensor of shape (batch_size, sequence_length, embedding_dim).

        Returns:
        - Output of attention mechanism applied to input `x`.
        """
        batch_size, seq_len, emb_dim = x.shape  # Renamed variables for clarity and snake_case
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) * emb_dim ** -0.5
        wei = wei.masked_fill(self.tril[:seq_len, :seq_len] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        return wei @ v


class MultiHeadAttention(nn.Module):
    """
    Multi-head self-attention mechanism that combines the outputs of several attention heads.

    Args:
    - num_heads (int): Number of attention heads.
    - head_size (int): The dimension of each attention head.
    - n_embd (int): The size of the input embedding.
    - block_size (int): The length of the sequence to consider.
    - dropout (float): Dropout rate to use after combining the outputs of the heads.

    Methods:
    - forward(x): Applies multi-head self-attention to the input `x` and returns the result.
    """
    def __init__(self, num_heads, head_size, n_embd, block_size, dropout):
        super().__init__()
        self.heads = nn.ModuleList(
            [Head(head_size, n_embd, block_size, dropout) for _ in range(num_heads)]
        )
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Performs forward pass of multi-head attention.

        Args:
        - x (Tensor): Input tensor of shape (batch_size, sequence_length, embedding_dim).

        Returns:
        - Output of multi-head attention applied to input `x`.
        """
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


class FeedFoward(nn.Module):
    """
    Feed-forward neural network layer used in transformer blocks.

    Args:
    - n_embd (int): The size of the input embedding.
    - dropout (float): Dropout rate to use in the feed-forward layers.

    Methods:
    - forward(x): Applies the feed-forward network to the input `x`.
    """
    def __init__(self, n_embd, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        """
        Performs forward pass of the feed-forward network.

        Args:
        - x (Tensor): Input tensor of shape (batch_size, sequence_length, embedding_dim).

        Returns:
        - Output of the feed-forward network.
        """
        return self.net(x)


class Block(nn.Module):
    """
    Transformer block consisting of multi-head self-attention and feed-forward layers.

    Args:
    - n_embd (int): The size of the input embedding.
    - n_head (int): The number of attention heads.
    - block_size (int): The length of the sequence to consider.
    - dropout (float): Dropout rate to use in the layers.

    Methods:
    - forward(x): Applies the transformer block to the input `x`.
    """
    def __init__(self, n_embd, n_head, block_size, dropout):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size, n_embd, block_size, dropout)
        self.ffwd = FeedFoward(n_embd, dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        """
        Performs forward pass of the transformer block.

        Args:
        - x (Tensor): Input tensor of shape (batch_size, sequence_length, embedding_dim).

        Returns:
        - Output of the transformer block applied to the input `x`.
        """
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class BigramLanguageModel(nn.Module):
    """
    Bigram-based language model with multiple transformer blocks for sequence generation.

    Args:
    - vocab_size (int): The size of the vocabulary.
    - n_embd (int): The size of the embedding vector for each token.
    - block_size (int): The length of the sequence to consider.
    - n_layer (int): The number of transformer blocks.
    - n_head (int): The number of attention heads in each transformer block.
    - dropout (float): Dropout rate to use in the layers.

    Methods:
    - forward(idx, targets=None): Computes the logits and optionally the loss for the input `idx`.
    - generate(idx, max_new_tokens, block_size): Generates a sequence of new tokens given an input sequence.
    """
    def __init__(self, vocab_size, n_embd, block_size, n_layer, n_head, dropout=0.0):
        """
        Args:
        - vocab_size (int): Size of the vocabulary.
        - n_embd (int): Embedding dimension.
        - block_size (int): Number of tokens in the input sequence.
        - n_layer (int): Number of transformer blocks.
        - n_head (int): Number of attention heads.
        - dropout (float, optional): Dropout probability (default is 0.0).
        """
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(
            *[Block(n_embd, n_head, block_size, dropout) for _ in range(n_layer)]
        )
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        """
        Performs the forward pass of the language model.

        Args:
        - idx (Tensor): Input tensor of token indices (batch_size, sequence_length).
        - targets (Tensor, optional): Target tensor for the input (batch_size, sequence_length).

        Returns:
        - logits (Tensor): The predicted logits for each token in the vocabulary.
        - loss (Tensor or None): Cross-entropy loss if targets are provided, otherwise None.
        """
        batch_size, seq_len = idx.shape  # Renamed variables to snake_case
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(
            torch.arange(seq_len, device=idx.device)
        )
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        if targets is None:
            loss = None
        else:
            logits = logits.view(batch_size * seq_len, -1)
            targets = targets.view(batch_size * seq_len)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_new_tokens=100, block_size=16):
        """
        Generates a sequence of tokens from the model.

        Args:
        - idx (Tensor): Input tensor of token indices (batch_size, sequence_length).
        - max_new_tokens (int, optional): Maximum number of new tokens to generate (default is 100).
        - block_size (int, optional): Number of tokens in the input sequence (default is 16).

        Returns:
        - idx (Tensor): Updated tensor of token indices after generation.
        """
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
