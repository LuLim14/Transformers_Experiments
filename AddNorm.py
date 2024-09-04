import torch
import torch.nn as nn

from torch import Tensor


class LayerNormalization(nn.Module):
    def __init__(self, embedding_dim: int, eps: float = 1e-6) -> None:
        super(LayerNormalization, self).__init__()
        self.layer_norm = nn.LayerNorm(normalized_shape=embedding_dim, eps=eps)

    def forward(self, x: Tensor) -> Tensor:
        return self.layer_norm(x)


class RMSNorm(nn.Module):
    def __init__(self, embedding_dim: int, eps: float = 1e-6) -> None:
        super(RMSNorm, self).__init__()
        self.embedding_dim = embedding_dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(self.embedding_dim))

    def _norm(self, x: Tensor) -> Tensor:
        return torch.rsqrt(torch.mean(x.pow(2), dim=-1, keepdim=True) + self.eps)

    def forward(self, x: Tensor) -> Tensor:
        rms = self._norm(x.float()).type_as(x)
        output = x * rms
        return output * self.weight


class FeedForwardLayer(nn.Module):
    def __init__(self, embedding_dim: int = 512, inner_dim: int = 2048) -> None:
        super(FeedForwardLayer, self).__init__()
        self.linear1 = nn.Linear(in_features=embedding_dim, out_features=inner_dim)
        self.activation = nn.ReLU()
        self.linear2 = nn.Linear(in_features=inner_dim, out_features=embedding_dim)

    def forward(self, x: Tensor) -> Tensor:
        x = self.activation(self.linear1(x))
        x = self.linear2(x)
        return x


class ResidualConnection(nn.Module):
    def __init__(self, embedding_dim: int, eps: float = 1e-05) -> None:
        super(ResidualConnection, self).__init__()
        self.layer_norm = LayerNormalization(embedding_dim=embedding_dim, eps=eps)

    def forward(self, x_input: Tensor, x_skip: Tensor) -> Tensor:
        return self.layer_norm(x_input + x_skip)

