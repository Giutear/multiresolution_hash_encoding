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
                 min_res: float = 0.05) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.dimension = dimension
        self.encoders = nn.ModuleList([
            MultiresolutionHashEncoding(hashtable_size,
                                        input_dim=3,
                                        feature_dim=2,
                                        levels=dimension // 2,
                                        N_min=1,
                                        N_max=int(1.0 / min_res))
            for _ in range(3 * num_heads)
        ])

    def forward(self, xyz: torch.Tensor):
        rel_xyz = (xyz.unsqueeze(1) - xyz.unsqueeze(0)).reshape(-1, 3)

        return tuple(
            torch.cat(tuple(self.encoders[i * self.num_heads + h](
                rel_xyz).reshape(1, xyz.shape[0], xyz.shape[0], -1)
                            for h in range(self.num_heads)),
                      dim=0) for i in range(3))


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
                 min_res: float = 0.05,
                 use_bias: bool = False) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.dimension = dimension // num_heads
        self.total_dim = dimension

        self.encoding = cRPEncoding(self.dimension, num_heads, hashtable_size,
                                    min_res)

        self.q = nn.Linear(self.total_dim, self.total_dim, bias=use_bias)
        self.kv = nn.Linear(self.total_dim, 2 * self.total_dim, bias=use_bias)
        self.proj = nn.Linear(self.total_dim, self.total_dim, bias=use_bias)

    def forward(self, xyz: torch.Tensor, feat: torch.Tensor):
        q_enc, k_enc, v_enc = self.encoding(xyz)
        k_enc: torch.Tensor
        queries = self.q(feat).reshape(self.num_heads, -1, self.dimension)
        kv: torch.Tensor = self.kv(feat)
        keys = kv[:, :self.total_dim].reshape(self.num_heads, -1,
                                              self.dimension)
        values = kv[:, self.total_dim:].reshape(self.num_heads, -1,
                                                self.dimension)

        attention = torch.matmul(queries, keys.swapaxes(-1, -2))

        attention += torch.matmul(queries.unsqueeze(-2),
                                  q_enc.swapaxes(-1, -2)).squeeze()
        attention += torch.matmul(keys.unsqueeze(-2),
                                  k_enc.permute(0, 2, 3,
                                                1)).squeeze().swapaxes(-1, -2)

        scores = torch.softmax(attention, dim=-1).unsqueeze(-1)

        y = ((values.unsqueeze(1) + v_enc) * scores).sum(dim=2).swapaxes(
            0, 1).reshape(-1, self.total_dim)
        return self.proj(y)
