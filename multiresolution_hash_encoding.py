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
    N_min: int
    N_max: int

    hash_tables: nn.Parameter
    resolutions: List[torch.Tensor]

    _prime_numbers: torch.Tensor
    _voxel_border_adds: torch.Tensor

    def __init__(self, hash_table_size: int, input_dim: int, feature_dim: int,
                 **kwargs):
        super().__init__()
        assert input_dim <= 7, "hash encoding only supports up to 7 dimensions."
        self.hash_table_size = hash_table_size
        self.input_dim = input_dim
        self.feature_dim = feature_dim
        self.levels = kwargs.get("levels")
        self.N_min = kwargs.get("N_min")
        self.N_max = kwargs.get("N_max")

        b = np.exp(
            (np.log(self.N_max) - np.log(self.N_min)) / (self.levels - 1))

        self.resolutions = nn.Parameter(
            torch.from_numpy(
                np.array([
                    np.floor(self.N_min * (b**i)) for i in range(self.levels)
                ],
                         dtype=np.int64)).reshape(1, 1, -1, 1), False)

        hash_table = torch.empty((self.levels, hash_table_size, feature_dim))
        self.hash_tables = nn.Parameter(hash_table)
        #Taken from nvidia's tiny cuda nn implementation
        self._prime_numbers = nn.Parameter(
            torch.from_numpy(
                np.array([
                    1, 2654435761, 805459861, 3674653429, 2097192037,
                    1434869437, 2165219737
                ])), False)

        border_adds = np.empty((self.input_dim, 2**self.input_dim),
                               dtype=np.int64)
        for i in range(self.input_dim):
            pattern = np.array(
                ([0] * (i + 1) + [1] * (i + 1)) * (self.input_dim // (i + 1)),
                dtype=np.int64)
            border_adds[i, :] = pattern
        self._voxel_border_adds = nn.Parameter(
            torch.from_numpy(border_adds).unsqueeze(0).unsqueeze(2), False)

    def forward(self, x: torch.Tensor):
        # 1. Scale and get surrounding grid coords
        scaled_coords = torch.mul(
            x.unsqueeze(-1).unsqueeze(-1), self.resolutions)
        grid_coords = torch.floor(scaled_coords).type(torch.int64)
        grid_coords = torch.add(grid_coords, self._voxel_border_adds)
        # 2. Hash the grid coords
        hashed_indices = self._fast_hash(grid_coords)
        # 3. Look up the hashed indices
        looked_up = torch.empty(
            (x.shape[0], self.feature_dim, self.levels, 2**self.input_dim),
            dtype=self.hash_tables.dtype,
            device=x.device)
        for i in range(x.shape[0]):
            for j in range(self.levels):
                looked_up[i, :, j] = self.hash_tables[j, hashed_indices[i,
                                                                        j]].T
        # 4. Interpolate features
        weights = torch.abs(
            torch.sub(scaled_coords, grid_coords.type(scaled_coords.dtype)))
        weights = torch.prod(weights, axis=1).unsqueeze(1)
        return torch.sum(torch.mul(weights, looked_up),
                         axis=-1).reshape(x.shape[0], -1)

    def _fast_hash(self, x: torch.Tensor) -> torch.Tensor:
        tmp = torch.zeros((x.shape[0], self.levels, 2**self.input_dim),
                          dtype=torch.int64,
                          device=x.device)
        for i in range(self.input_dim):
            tmp = torch.bitwise_xor(x[:, i, :, :] * self._prime_numbers[i],
                                    tmp)
        return torch.remainder(
            tmp, torch.tensor(self.hash_table_size, dtype=torch.int64))
