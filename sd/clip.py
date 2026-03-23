import torch
from torch import nn
from torch.nn import functional as F
from attention import SelfAttention


class CLIPEmbedding(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int, max_seq_len: int):
        super().__init__()

        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.position_embedding = nn.Parameter(torch.zeros(max_seq_len, embedding_dim))

    def forward(self, tokens: torch.LongTensor) -> torch.Tensor:
        # tokens: (Batch, Seq_Len) -> (Batch, Seq_Len, Dim)
        x = self.token_embedding(tokens)

        x += self.position_embedding

        return x


class CLIPLayer(nn.Module):
    def __init__(self, num_heads: int, embedding_dim: int):
        super().__init__()

        self.layernorm_1 = nn.LayerNorm(embedding_dim)
        self.attention = SelfAttention(num_heads, embedding_dim)

        self.layernorm_2 = nn.LayerNorm(embedding_dim)

        self.linear_1 = nn.Linear(embedding_dim, embedding_dim * 4)
        self.linear_2 = nn.Linear(embedding_dim * 4, embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (Batch, Seq_Len, Dim)
        residue = x

        x = self.layernorm_1(x)
        x = self.attention(x, causal_mask=True)
        x += residue

        residue = x

        x = self.layernorm_2(x)
        x = self.linear_1(x)
        x = x * torch.sigmoid(x * 1.702)
        x = self.linear_2(x)

        x += residue

        return x


class CLIP(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = CLIPEmbedding(49408, 768, 77)

        self.layers = nn.ModuleList([CLIPLayer(12, 768) for i in range(12)])

        self.layernorm = nn.LayerNorm(768)

    def forward(self, tokens: torch.LongTensor) -> torch.FloatTensor:
        tokens = tokens.type(torch.long)

        # (Batch_Size, Seq_Len) -> (Batch_Size, Seq_Len, Dim)
        state = self.embedding(tokens)

        for layer in self.layers:
            # (Batch_Size, Seq_Len, Dim)
            # -> (Batch_Size, Seq_Len, Dim)
            state = layer(state)
        # (Batch_Size, Seq_Len, Dim)
        # -> (Batch_Size, Seq_Len, Dim)
        output = self.layernorm(state)

        return output
