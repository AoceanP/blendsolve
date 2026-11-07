"""The neural blendshape solver and the least-squares baseline it is measured against.

WHAT: Solver is an MLP that maps aligned vertex offsets to weights in [0, 1].
      least_squares() solves the linear part of the rig directly, as a classical baseline.
WHY:  a linear solve cannot model the rotating jaw or ignore outliers well; comparing both
      shows what the network actually adds.
"""
import torch
from torch import nn

from . import rig


class Solver(nn.Module):
    """MLP solver: (B, N*3) offsets -> (B, K+1) weights through a sigmoid."""

    def __init__(self, n_in: int, n_out: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_in, hidden), nn.GELU(), nn.LayerNorm(hidden),
                                 nn.Linear(hidden, hidden // 2), nn.GELU(),
                                 nn.Linear(hidden // 2, n_out))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(x))


def least_squares(x: torch.Tensor) -> torch.Tensor:
    """Baseline: solve offsets = A w with jawOpen linearized at w = 0, then clamp to [0, 1].

    RETURN: (B, K+1) weights.
    """
    base = rig.neutral()
    d = rig.deltas(base)
    eps = 1e-3                                                          # numeric jaw derivative
    zero = torch.zeros(1, len(rig.NAMES))
    jaw = zero.clone()
    jaw[0, -1] = eps
    jaw_col = ((rig.pose(jaw, base, d) - rig.pose(zero, base, d)) / eps).reshape(-1)
    a = torch.cat([d.reshape(d.shape[0], -1), jaw_col[None]], 0).T     # (N*3, K+1)
    w = torch.linalg.lstsq(a, x.T).solution.T
    return w.clamp(0, 1)
