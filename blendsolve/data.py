"""Synthetic "capture" data: posed meshes damaged the way raw scans are, plus the cleanup step.

WHAT: samples sparse expression weights, poses the rig, then adds a random head transform,
      sensor jitter and outlier vertices. clean() undoes the head transform with Kabsch.
WHY:  the model should learn from noisy captures, and rigid alignment is the standard first
      cleanup step before solving for blendshapes.
"""
import math

import torch

from . import rig


def sample_weights(n: int, g: torch.Generator) -> torch.Tensor:
    """Return (n, K+1) weights: each shape is active with probability 0.35, value in [0, 1]."""
    k = len(rig.NAMES)
    active = torch.rand(n, k, generator=g) < 0.35
    return torch.rand(n, k, generator=g) * active


def random_rotation(n: int, max_deg: float, g: torch.Generator) -> torch.Tensor:
    """Return (n, 3, 3) rotation matrices from random axes and angles up to max_deg."""
    axis = torch.nn.functional.normalize(torch.randn(n, 3, generator=g), dim=1)
    ang = (torch.rand(n, generator=g) * 2 - 1) * math.radians(max_deg)
    kx = torch.zeros(n, 3, 3)
    kx[:, 0, 1], kx[:, 0, 2], kx[:, 1, 2] = -axis[:, 2], axis[:, 1], -axis[:, 0]
    kx = kx - kx.transpose(1, 2)
    eye = torch.eye(3).expand(n, 3, 3)
    s, c = torch.sin(ang)[:, None, None], torch.cos(ang)[:, None, None]
    return eye + s * kx + (1 - c) * (kx @ kx)                            # Rodrigues formula


def capture(verts: torch.Tensor, g: torch.Generator, noise: float = 0.01,
            outlier_frac: float = 0.02) -> torch.Tensor:
    """Damage clean meshes (B, N, 3): rotate up to 15 degrees, translate, jitter, add outliers."""
    b, n, _ = verts.shape
    r = random_rotation(b, 15.0, g)
    t = (torch.rand(b, 1, 3, generator=g) * 2 - 1) * 0.3
    out = verts @ r.transpose(1, 2) + t + torch.randn(b, n, 3, generator=g) * noise
    bad = torch.rand(b, n, generator=g) < outlier_frac
    out[bad] += torch.randn(int(bad.sum()), 3, generator=g) * 0.15
    return out


def kabsch(src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
    """Rigidly align each src (B, N, 3) onto dst (N, 3) with the SVD-based Kabsch algorithm.

    RETURN: aligned src, same shape. Reflections are corrected so the result is a rotation.
    """
    sc, dc = src.mean(1, keepdim=True), dst.mean(0, keepdim=True)
    h = (src - sc).transpose(1, 2) @ (dst - dc)                          # (B, 3, 3)
    u, _, vt = torch.linalg.svd(h)
    sign = torch.sign(torch.linalg.det(vt.transpose(1, 2) @ u.transpose(1, 2)))
    fix = torch.diag_embed(torch.stack([torch.ones_like(sign), torch.ones_like(sign), sign], 1))
    r = vt.transpose(1, 2) @ fix @ u.transpose(1, 2)
    return (src - sc) @ r.transpose(1, 2) + dc


def make_dataset(n: int, seed: int, noise: float = 0.01):
    """Return (features, weights): aligned capture offsets from neutral (n, N*3) and targets."""
    g = torch.Generator().manual_seed(seed)
    base, d = rig.neutral(), rig.deltas(rig.neutral())
    w = sample_weights(n, g)
    raw = capture(rig.pose(w, base, d), g, noise=noise)
    x = (kabsch(raw, base) - base).reshape(n, -1)
    return x, w
