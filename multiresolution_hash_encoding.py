"""Implements the Multiresolution Hash Encoding by NVIDIA."""
from typing import List
import torch
from torch import nn
import numpy as np


class MultiresolutionHashEncoding(nn.Module):
    hash_table_size: int
    input_dim: int
    feature_dim: int
    levels: int

    hash_tables: List[nn.Parameter]
    resolutions: List[torch.Tensor]

    _prime_numbers: torch.Tensor

    def __init__(self, hash_table_size: int, input_dim: int, feature_dim: int,
                 **kwargs):
        super().__init__()
        assert input_dim <= 7, "hash encoding only supports up to 7 dimensions."
        self.hash_table_size = hash_table_size
        self.input_dim = input_dim
        self.feature_dim = feature_dim
        self.levels = kwargs.get("levels")

        for _ in range(self.levels):
            hash_table = torch.empty((hash_table_size, feature_dim))
            self.hash_tables.append(nn.Parameter(hash_table))
        #Taken from nvidia's tiny cuda nn implementation
        self._prime_numbers = torch.tensor(np.array([
            1, 2654435761, 805459861, 3674653429, 2097192037, 1434869437,
            2165219737
        ]),
                                           dtype=torch.int64)

    def forward(self, x: torch.Tensor):
        pass

    def _fast_hash(self, x: torch.Tensor) -> torch.Tensor:
        tmp = torch.zeros((x.shape[0], 4))
        for i in range(self.input_dim):
            tmp = torch.bitwise_xor(x[:, :, i], tmp)
        return torch.remainder(
            tmp, torch.tensor(self.hash_table_size, dtype=torch.int64))
