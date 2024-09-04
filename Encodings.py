import torch
import torch.nn as nn
import math

from torch import Tensor


class PositionalEncoding(nn.Module):
    def __init__(self, max_sequence_length: int, embedding_dim: int) -> None:
        super(PositionalEncoding, self).__init__()
        self.embedding_dim = embedding_dim
        self.max_sequence_length = max_sequence_length
        positions = torch.arange(0, self.max_sequence_length, 2, dtype=torch.int32)
        powers = torch.exp(-math.log(10000) / self.embedding_dim * positions.float())
        self.positional_encoding = torch.zeros(self.max_sequence_length, self.embedding_dim)
        self.positional_encoding[:, 0::2] = torch.sin(positions[:, 0::2] * powers)
        self.positional_encoding[:, 1::2] = torch.cos(positions[:, 1::2] * powers)
        self.positional_encoding = self.positional_encoding.unsqueeze(0)
        self.register_buffer('positional_encoding', self.positional_encoding)
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x: Tensor) -> Tensor:
        """"
        Args:
            x: Tensor of shape (seq_len, batch_size, d_model)
        Returns:
            Tensor with positional encoding added to input with shape (seq_len, batch_size, d_model).
        """
        x = x + self.positional_encoding[:, x.size(0), :]
        return self.dropout(x)




# add rotary embeddings