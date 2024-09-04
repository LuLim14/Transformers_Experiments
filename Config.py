from dataclasses import dataclass


@dataclass
class Config:
    pass


@dataclass
class ModelArgs:
    embedding_dim: int = 512
    num_layers_encoder: int = 2
    num_layers_decoder: int = 2
    num_heads: int = 8