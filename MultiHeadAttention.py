import math
import torch.nn as nn

from torch import Tensor


class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, embedding_dim: int, num_heads: int) -> None:
        super(MultiHeadAttentionLayer, self).__init__()
        assert(embedding_dim % num_heads == 0)

        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = self.embedding_dim // self.num_heads

        self.W_q = nn.Linear(in_features=self.embedding_dim, out_features=self.embedding_dim)
        self.W_k = nn.Linear(in_features=self.embedding_dim, out_features=self.embedding_dim)
        self.W_v = nn.Linear(in_features=self.embedding_dim, out_features=self.embedding_dim)

        self.W_o = nn.Linear(in_features=self.embedding_dim, out_features=self.embedding_dim)

    @staticmethod
    def attention_score(query: Tensor, key: Tensor, value: Tensor, mask=None) -> Tensor:
        attention_score = query @ key.T / math.sqrt(query.shape[-1])
        if mask is not None:
            attention_score = attention_score.masked_fill(mask == 0, value=1e-9)
        attention_probs = attention_score.softmax(dim=-1)
        output = attention_probs @ value
        return output

    @staticmethod
    def combine_heads(input_embedding: Tensor) -> Tensor:
        batch_size, _, seq_length, embedding_dim = input_embedding.size()
        return input_embedding.transpose(1, 2).contiguous().view(batch_size, seq_length, -1)

    def forward(self, q: Tensor, k: Tensor, v: Tensor, mask=None) -> Tensor:
        q = self.W_q(q)
        k = self.W_k(k)
        v = self.W_v(v)

        attention_output = MultiHeadAttentionLayer.attention_score(query=q, key=k, value=v, mask=mask)

        attention_output_concat = MultiHeadAttentionLayer.combine_heads(attention_output)

        return self.W_o(attention_output_concat)

