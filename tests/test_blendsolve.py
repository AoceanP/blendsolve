"""Unit tests for the rig, the capture/cleanup pipeline and the solver."""
import math

import torch

from blendsolve import data, model, rig


def test_zero_weights_give_neutral():
    base, d = rig.neutral(), rig.deltas(rig.neutral())
    out = rig.pose(torch.zeros(1, len(rig.NAMES)), base, d)
    assert torch.allclose(out[0], base, atol=1e-6)


def test_blendshapes_are_local():
    base = rig.neutral()
    d = rig.deltas(base)
    left = rig.SHAPES.index("browRaise_L")
    far = base[:, 0] > 0.5                                             # right side of the face
    assert d[left][far].norm(dim=-1).max() < 1e-2


def test_jaw_moves_lower_face_only():
    base, d = rig.neutral(), rig.deltas(rig.neutral())
    w = torch.zeros(1, len(rig.NAMES))
    w[0, -1] = 1.0
    move = (rig.pose(w, base, d)[0] - base).norm(dim=-1)
    assert move[base[:, 1] < -0.8].mean() > 0.1
    assert move[base[:, 1] > 0.5].max() < 1e-2


def test_rotation_is_orthonormal():
    r = data.random_rotation(8, 30.0, torch.Generator().manual_seed(0))
    eye = torch.eye(3).expand(8, 3, 3)
    assert torch.allclose(r @ r.transpose(1, 2), eye, atol=1e-5)
    assert torch.allclose(torch.linalg.det(r), torch.ones(8), atol=1e-5)


def test_kabsch_undoes_rigid_transform():
    base = rig.neutral()
    g = torch.Generator().manual_seed(1)
    r = data.random_rotation(4, 40.0, g)
    moved = base[None] @ r.transpose(1, 2) + torch.tensor([0.5, -0.2, 0.3])
    assert torch.allclose(data.kabsch(moved, base), base.expand(4, -1, -1), atol=1e-4)


def test_capture_adds_noise_and_outliers():
    base, d = rig.neutral(), rig.deltas(rig.neutral())
    g = torch.Generator().manual_seed(2)
    clean = rig.pose(torch.zeros(2, len(rig.NAMES)), base, d)
    aligned = data.kabsch(data.capture(clean, g), base)
    err = (aligned - base).norm(dim=-1)
    assert err.mean() > 0.005 and err.max() > 0.1


def test_dataset_shapes_and_range():
    x, w = data.make_dataset(16, seed=0)
    assert x.shape == (16, rig.GRID * rig.GRID * 3)
    assert w.shape == (16, len(rig.NAMES))
    assert w.min() >= 0 and w.max() <= 1


def test_least_squares_recovers_clean_linear_weights():
    base, d = rig.neutral(), rig.deltas(rig.neutral())
    w = torch.zeros(1, len(rig.NAMES))
    w[0, 0], w[0, 4] = 0.7, 0.3
    x = (rig.pose(w, base, d) - base).reshape(1, -1)
    assert torch.allclose(model.least_squares(x), w, atol=1e-3)


def test_solver_outputs_valid_weights():
    net = model.Solver(30, 5)
    out = net(torch.randn(7, 30))
    assert out.shape == (7, 5) and out.min() >= 0 and out.max() <= 1


def test_solver_learns_quickly():
    torch.manual_seed(0)
    x, w = data.make_dataset(512, seed=4)
    mu, sd = x.mean(0), x.std(0) + 1e-6
    net = model.Solver(x.shape[1], w.shape[1], hidden=64)
    opt = torch.optim.Adam(net.parameters(), lr=3e-3)
    first = torch.nn.functional.l1_loss(net((x - mu) / sd), w).item()
    for _ in range(60):
        loss = torch.nn.functional.l1_loss(net((x - mu) / sd), w)
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < first * 0.6
    assert not math.isnan(loss.item())
