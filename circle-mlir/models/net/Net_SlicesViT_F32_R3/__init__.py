import torch
from torch import nn
from einops import rearrange


# Generate simplified network of 'SliceOp's from Attention part of Transformer
class SlicesViT(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64):
        super().__init__()

        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head
        self.norm = nn.LayerNorm(dim)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)
        return q + k + v


_model_ = SlicesViT(2, 2, 2)

_inputs_ = torch.randn(2, 2, 2)
