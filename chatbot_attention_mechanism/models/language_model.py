import torch
from torch import nn as nn
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
        - x (Tensor): Input tensor of shape (batch_size, seq_length, embedding_dim).

        Returns:
        - Output of attention mechanism applied to input `x`.
        """
        batch_size, seq_length, embed_dim = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) * embed_dim ** -0.5
        wei = wei.masked_fill(self.tril[:seq_length, :seq_length] == 0, float('-inf'))
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
        - x (Tensor): Input tensor of shape (batch_size, seq_length, embedding_dim).

        Returns:
        - Output of multi-head attention applied to input `x`.
        """
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


class FeedForward(nn.Module):
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
        - x (Tensor): Input tensor of shape (batch_size, seq_length, embedding_dim).

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
        self.ffwd = FeedForward(n_embd, dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        """
        Performs forward pass of the transformer block.

        Args:
        - x (Tensor): Input tensor of shape (batch_size, seq_length, embedding_dim).

        Returns:
        - Output of the transformer block applied to the input `x`.
        """
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x
