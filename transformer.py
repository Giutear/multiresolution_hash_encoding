import torch
from torch import nn
from attention import ContextMultiheadAttention


class ContextTransformer(nn.Module):

    attention: ContextMultiheadAttention
    pre_norm: nn.LayerNorm
    post_transform: nn.ModuleList

    def __init__(self,
                 f_dim: int,
                 num_heads: int,
                 min_res: float,
                 hashtable_size: int = 2**12) -> None:
        super().__init__()
        self.attention = ContextMultiheadAttention(f_dim, num_heads,
                                                   hashtable_size, min_res)
        self.pre_norm = nn.LayerNorm(f_dim)
        self.post_transform = nn.ModuleList(
            [nn.LayerNorm(f_dim), nn.Linear(f_dim, f_dim)])

    def forward(self, xyz: torch.Tensor, feat: torch.Tensor):
        feat = self.pre_norm(feat)
        feat = feat + self.attention(xyz, feat)
        _feat = feat
        for layer in self.post_transform:
            _feat = layer(_feat)
        return _feat + feat