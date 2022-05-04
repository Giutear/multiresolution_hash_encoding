"""Implements the Multiresolution Hash Encoding by NVIDIA."""
import torch
from torch import nn
import numpy as np


class MultiresolutionHashEncoding(nn.Module):
    '''
    Implements the Multiresolution Hash Encoding by NVIDIA.
    '''
    hash_table_size: int
    input_dim: int
    feature_dim: int
    levels: int
    N_min: int
    N_max: int

    _hash_tables: nn.Parameter
    _resolutions: nn.Parameter

    _prime_numbers: nn.Parameter
    _voxel_border_adds: nn.Parameter

    def __init__(self, hash_table_size: int, input_dim: int, feature_dim: int,
                 **kwargs):
        '''
        Args:
            hash_table_size: The size of the hash tables.
            input_dim: The dimension of the input dim. Must be less or equal to 7.
            feature_dim: The dimension of the stored features per level.
            levels: The number of resolutionis that will be generated.
            N_min: The minimum resolution.
            N_max: The maximum resolution.
        '''
        super().__init__()
        assert input_dim <= 7, "hash encoding only supports up to 7 dimensions."
        self.hash_table_size = hash_table_size
        self.input_dim = input_dim
        self.feature_dim = feature_dim
        self.levels = kwargs.get("levels")
        self.N_min = kwargs.get("N_min")
        self.N_max = kwargs.get("N_max")

        # Calculate resolution as stated in the paper
        b = np.exp(
            (np.log(self.N_max) - np.log(self.N_min)) / (self.levels - 1))

        self._resolutions = nn.Parameter(
            torch.from_numpy(
                np.array([
                    np.floor(self.N_min * (b**i)) for i in range(self.levels)
                ],
                         dtype=np.int64)).reshape(1, 1, -1, 1), False)

        hash_table = torch.empty((self.levels, hash_table_size, feature_dim))
        self._hash_tables = nn.Parameter(hash_table)
        nn.init.uniform_(self._hash_tables, -10.0**(-4), 10.0**(-4))
        #Taken from nvidia's tiny cuda nn implementation
        self._prime_numbers = nn.Parameter(
            torch.from_numpy(
                np.array([
                    1, 2654435761, 805459861, 3674653429, 2097192037,
                    1434869437, 2165219737
                ])), False)
        # This is a helper tensor which generates the voxel coordinates.
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
        '''
        Takes a set of input vectors and encodes them.

        Args:
            x: A tensor of the shape (batch, input_dim) of all input vectors.

        Returns:
            A tensor of the shape (batch, levels * feature_dim)
            containing the encoded input vectors.
        '''
        # 1. Scale and get surrounding grid coords
        scaled_coords = torch.mul(
            x.unsqueeze(-1).unsqueeze(-1), self._resolutions)
        if scaled_coords.dtype in [torch.float32, torch.float64]:
            grid_coords = torch.floor(scaled_coords).type(torch.int64)
        else:
            grid_coords = scaled_coords
        grid_coords = torch.add(grid_coords, self._voxel_border_adds)
        # 2. Hash the grid coords
        hashed_indices = self._fast_hash(grid_coords)
        # 3. Look up the hashed indices
        looked_up = torch.empty(
            (x.shape[0], self.feature_dim, self.levels, 2**self.input_dim),
            dtype=self._hash_tables.dtype,
            device=x.device,
            requires_grad=True)
        for j in range(self.levels):
            looked_up[:, :,
                      j] = self._hash_tables[j, hashed_indices[:, j]].permute(
                          0, 2, 1)
        # 4. Interpolate features
        weights = torch.abs(
            torch.sub(scaled_coords, grid_coords.type(scaled_coords.dtype)))
        weights = torch.prod(weights, axis=1).unsqueeze(1)
        return torch.sum(torch.mul(weights, looked_up),
                         axis=-1).reshape(x.shape[0], -1)

    def _fast_hash(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Implements the hash function proposed by NVIDIA.

        Args:
            x: A tensor of the shape (batch, input_dim, levels, 2^input_dim).
               This tensor should contain the vertices of the hyper cuber
               for each level.

        Returns:
            A tensor of the shape (batch, levels, 2^input_dim) containing the
            indices into the hash table for all vertices.
        '''
        tmp = torch.zeros((x.shape[0], self.levels, 2**self.input_dim),
                          dtype=torch.int64,
                          device=x.device)
        for i in range(self.input_dim):
            tmp = torch.bitwise_xor(x[:, i, :, :] * self._prime_numbers[i],
                                    tmp)
        return torch.remainder(
            tmp, torch.tensor(self.hash_table_size, dtype=torch.int64))
