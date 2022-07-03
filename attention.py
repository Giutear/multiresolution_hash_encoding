import torch
from torch import nn
from multiresolution_hash_encoding import MultiresolutionHashEncoding


class cRPEncoding(nn.Module):

    encoders: nn.ModuleList
    num_heads: int

    def __init__(self,
                 dimension: int,
                 num_heads: int,
                 hashtable_size: int = 2**12,
                 res: float = 1e7) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.dimension = dimension
        self.encoders = nn.ModuleList([
            MultiresolutionHashEncoding(hashtable_size,
                                        input_dim=3,
                                        feature_dim=2,
                                        levels=dimension // 2,
                                        N_min=1,
                                        N_max=res)
            for _ in range(3 * num_heads)
        ])

    def forward(self, xyz: torch.Tensor, edges: torch.LongTensor):
        rel_xyz = (xyz.unsqueeze(-2) - xyz[edges]).reshape(-1, 3)

        return tuple(
            torch.cat(tuple(self.encoders[i * self.num_heads + h](
                rel_xyz).reshape(xyz.shape[0], 1, edges.shape[1], -1)
                            for h in range(self.num_heads)),
                      dim=1) for i in range(3))


class ContextMultiheadAttention(nn.Module):

    encoding: cRPEncoding
    q: nn.Linear
    kv: nn.Linear
    proj: nn.Linear

    num_heads: int

    def __init__(self,
                 dimension: int,
                 num_heads: int,
                 hashtable_size: int = 2**12,
                 res: float = 1e7,
                 use_bias: bool = False) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.dimension = dimension // num_heads
        self.total_dim = dimension

        self.encoding = cRPEncoding(self.dimension, num_heads, hashtable_size,
                                    res)

        self.q = nn.Linear(self.total_dim, self.total_dim, bias=use_bias)
        self.kv = nn.Linear(self.total_dim, 2 * self.total_dim, bias=use_bias)
        self.proj = nn.Linear(self.total_dim, self.total_dim, bias=use_bias)

    def forward(self, xyz: torch.Tensor, feat: torch.Tensor, edges):
        k = edges.shape[1]

        q_enc, k_enc, v_enc = self.encoding(xyz, edges)
        k_enc: torch.Tensor
        queries = self.q(feat).reshape(-1, self.num_heads, 1, self.dimension)
        kv: torch.Tensor = self.kv(feat)
        keys = kv[:, :self.total_dim][edges].reshape(-1, k, self.num_heads,
                                                     self.dimension).swapaxes(
                                                         -2, -3)
        values = kv[:, self.total_dim:][edges].reshape(
            -1, k, self.num_heads, self.dimension).swapaxes(-2, -3)

        attention = torch.matmul(queries, keys.swapaxes(-1, -2))

        attention += torch.matmul(queries, q_enc.swapaxes(-1, -2))
        attention += torch.matmul(keys.unsqueeze(-2),
                                  k_enc.unsqueeze(-1)).squeeze().unsqueeze(-2)

        scores = torch.softmax(attention, dim=-1).swapaxes(-1, -2)

        y = ((values + v_enc) * scores).sum(dim=2).reshape(-1, self.total_dim)
        return self.proj(y)
