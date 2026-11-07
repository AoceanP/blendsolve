"""A small procedural face rig: a neutral mesh plus blendshapes and one joint-driven jaw.

WHAT: builds a grid mesh shaped like a face patch, 11 localized blendshape targets and a jaw
      that rotates the lower vertices, then poses the mesh from a weight vector.
WHY:  real rigs mix linear blendshapes with nonlinear joint deformation; the jaw rotation
      makes the inverse problem nonlinear, which is what the neural solver has to learn.
"""
import math

import torch

GRID = 24  # 24 x 24 = 576 vertices
SHAPES = ["browRaise_L", "browRaise_R", "blink_L", "blink_R", "smile_L", "smile_R",
          "cheekPuff_L", "cheekPuff_R", "noseSneer", "lipPucker", "lipPress"]
JAW = "jawOpen"
NAMES = SHAPES + [JAW]
JAW_MAX_DEG = 25.0


def neutral() -> torch.Tensor:
    """Return the neutral mesh as (N, 3): a curved patch, x across, y up, z toward the camera.

    POST: tensor of shape (GRID*GRID, 3) in roughly [-1, 1] units.
    """
    u = torch.linspace(-1.0, 1.0, GRID)
    v = torch.linspace(-1.2, 1.2, GRID)
    yy, xx = torch.meshgrid(v, u, indexing="ij")
    zz = 0.6 * torch.cos(xx * math.pi / 2.4) * torch.cos(yy * math.pi / 3.0)
    return torch.stack([xx, yy, zz], dim=-1).reshape(-1, 3)


def _bump(base: torch.Tensor, cx: float, cy: float, r: float, disp: tuple) -> torch.Tensor:
    """Return a smooth Gaussian displacement field centred at (cx, cy) with peak vector disp."""
    d2 = (base[:, 0] - cx) ** 2 + (base[:, 1] - cy) ** 2
    return torch.exp(-d2 / (2 * r * r))[:, None] * torch.tensor(disp)


def deltas(base: torch.Tensor) -> torch.Tensor:
    """Return the blendshape delta matrix (K, N, 3) for the linear shapes in SHAPES order."""
    spec = [(-0.45, 0.75, 0.18, (0, 0.12, 0.02)), (0.45, 0.75, 0.18, (0, 0.12, 0.02)),
            (-0.4, 0.45, 0.10, (0, -0.06, -0.03)), (0.4, 0.45, 0.10, (0, -0.06, -0.03)),
            (-0.35, -0.55, 0.15, (-0.08, 0.08, 0.0)), (0.35, -0.55, 0.15, (0.08, 0.08, 0.0)),
            (-0.5, -0.2, 0.2, (0, 0, 0.12)), (0.5, -0.2, 0.2, (0, 0, 0.12)),
            (0.0, 0.05, 0.12, (0, 0.05, 0.03)), (0.0, -0.6, 0.12, (0, 0, 0.1)),
            (0.0, -0.6, 0.14, (0, 0.03, -0.04))]
    return torch.stack([_bump(base, *s) for s in spec])


def pose(weights: torch.Tensor, base: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
    """Pose the mesh: linear blendshapes, then rotate vertices below the mouth by the jaw angle.

    PRE:  weights (B, K+1) in [0, 1], last column is jawOpen.
    RETURN: posed vertices (B, N, 3).
    """
    lin = base[None] + torch.einsum("bk,knc->bnc", weights[:, :-1], d)
    angle = weights[:, -1] * math.radians(JAW_MAX_DEG)                  # (B,)
    pivot = torch.tensor([0.0, -0.1, -0.4])
    mask = torch.sigmoid((-0.35 - base[:, 1]) * 12.0)[None, :, None]   # soft lower-face region
    c, s = torch.cos(angle)[:, None], torch.sin(angle)[:, None]
    p = lin - pivot
    y = p[..., 1] * c - p[..., 2] * s                                   # rotate about the x axis
    z = p[..., 1] * s + p[..., 2] * c
    rot = torch.stack([p[..., 0], y, z], dim=-1) + pivot
    return lin * (1 - mask) + rot * mask
