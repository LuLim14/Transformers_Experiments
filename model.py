import torch.nn as nn

from typing import Type
from MultiHeadAttention import MultiHeadAttentionLayer
from AddNorm import FeedForwardLayer
from Config import ModelArgs


class EncoderBlock(nn.Module):
    def __init__(self, self_attention_block: Type[MultiHeadAttentionLayer],
                 feed_forward_block: Type[FeedForwardLayer], args: Type[ModelArgs]) -> None:
        super(EncoderBlock, self).__init__()
        self.embedding_dim = args.embedding_dim
        self.num_heads = args.num_heads
        self.head_dim = self.embedding_dim // self.num_heads

        self.self_attention_block = self_attention_block(embedding_dim=self.embedding_dim, num_heads=self.num_heads)
        self.feed_forward_block = feed_forward_block(embedding_dim=self.embedding_dim, inner_dim=4 * self.embedding_dim)
         #Residual Connection etc.

