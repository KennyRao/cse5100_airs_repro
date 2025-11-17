# airs/intrinsic.py
import math
from typing import Optional

import torch


class IdentityIntrinsic:
    """
    ID intrinsic: always returns zero.
    """
    def __init__(self, device: torch.device):
        self.device = device

    def reset(self):
        pass

    @torch.no_grad()
    def compute(self, obs_tensor: torch.Tensor) -> torch.Tensor:
        """
        obs_tensor: [B, C, H, W] (unused)
        returns: [B] zeros
        """
        return torch.zeros(obs_tensor.shape[0], device=self.device)


class RE3Intrinsic:
    """
    Minimal RE3 implementation:
    - Random, fixed encoder (passed in)
    - Buffer of past embeddings
    - Intrinsic reward: average log distance to k nearest neighbors
      I_t = 1/k sum_i log(||e_t - e_i||^2 + 1)
    """
    def __init__(
        self,
        encoder,
        device: torch.device,
        k: int = 3,
        max_buffer_size: int = 10000
    ):
        self.encoder = encoder
        self.device = device
        self.k = k
        self.max_buffer_size = max_buffer_size
        self._buffer: Optional[torch.Tensor] = None  # [N, D]

    def reset(self):
        self._buffer = None

    @torch.no_grad()
    def compute(self, obs_tensor: torch.Tensor) -> torch.Tensor:
        """
        obs_tensor: [B, C, H, W]
        returns: [B] intrinsic rewards
        """
        # Encode observations
        z = self.encoder(obs_tensor)  # [B, D]

        if self._buffer is None or self._buffer.shape[0] == 0:
            # No neighbors yet
            intrinsic = torch.zeros(z.shape[0], device=self.device)
            self._buffer = z.detach().clone()
            return intrinsic

        # Distances to buffer [B, N]
        # For MiniGrid + small buffer, cdist is okay.
        dists = torch.cdist(z, self._buffer)  # l2 distances

        # Use k smallest distances
        k_eff = min(self.k, dists.shape[1])
        knn_dists, _ = torch.topk(dists, k=k_eff, largest=False, dim=1)
        intrinsic = torch.log(knn_dists.pow(2) + 1.0).mean(dim=1)

        # Update buffer
        new_buffer = torch.cat([self._buffer, z.detach()], dim=0)
        if new_buffer.shape[0] > self.max_buffer_size:
            new_buffer = new_buffer[-self.max_buffer_size:]
        self._buffer = new_buffer

        return intrinsic


def beta_schedule(global_step: int, beta0: float, kappa: float) -> float:
    """
    β_t = β_0 * (1 - κ)^t
    For MiniGrid in AIRS: κ = 0 (no decay) and different β_0 per env.
    """
    return float(beta0 * math.pow(1.0 - kappa, global_step))
