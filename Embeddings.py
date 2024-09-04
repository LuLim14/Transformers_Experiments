import torch.nn as nn

from torch import Tensor


class Embeddings(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int) -> None:
        super(Embeddings, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(num_embeddings=self.vocab_size, embedding_dim=self.embedding_dim)

    def forward(self, input_tokens) -> Tensor:  # type?
        embed = self.embedding(input_tokens)
        return embed
