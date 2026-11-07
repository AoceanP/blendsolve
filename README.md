# blendsolve

A PyTorch blendshape solver for noisy 3D face captures. Given a raw, misaligned, noisy scan of a
face mesh, it predicts the 12 rig weights (11 blendshapes plus a joint-driven jaw) that produced it.

## Why

Facial pipelines constantly need the inverse of a rig: "which expression weights make this
capture?" A linear least-squares solve is the classic answer, but real rigs are not purely linear
(the jaw rotates) and real captures have head motion, sensor jitter and bad vertices. This project
measures how much a small neural network improves on the classical solve under those conditions.

## How it works

1. **Rig** (`blendsolve/rig.py`): a 576-vertex face patch with 11 localized Gaussian blendshapes
   (brows, blinks, smiles, cheeks, nose, lips) and a jaw that rotates the lower face up to 25 degrees.
2. **Capture** (`blendsolve/data.py`): each posed mesh gets a random head rotation (up to 15
   degrees) and translation, per-vertex jitter, and 2% outlier vertices.
3. **Cleanup**: the Kabsch algorithm (SVD) rigidly aligns every capture back to the neutral pose.
4. **Solve** (`blendsolve/model.py`): an MLP maps the aligned vertex offsets to weights in [0, 1].
   The baseline solves the same offsets with least squares, linearizing the jaw at rest.

All data is generated, so the project needs no downloads beyond PyTorch.

## Results

Held-out test set of 2,000 noisy captures (trained on 20,000, CPU):

| Metric | Neural solver | Least squares | Improvement |
|---|---|---|---|
| Mean vertex error of the re-posed mesh | 0.0041 | 0.0224 | 5.5x lower |
| Jaw weight error | 0.017 | 0.113 | 6.8x lower |
| Mean weight error (all 12) | 0.089 | 0.161 | 1.8x lower |

Training takes about 10 seconds on a laptop CPU, and inference takes 0.08 ms per frame
(478k parameters). Exact numbers are written to `results.json` by every run.

Small blendshapes such as blinks move vertices by about the same amount as the sensor noise, which
is why the per-weight error stays higher than the vertex error.

## Run it

```bash
python -m venv .venv
.venv/Scripts/python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv/Scripts/python -m pip install numpy pytest
.venv/Scripts/python -m pytest -q tests
.venv/Scripts/python train.py
```

Or with Docker, which runs the tests and then trains:

```bash
docker build -t blendsolve .
docker run --rm blendsolve
```

## Tests

10 pytest tests cover the rig (neutral pose, locality, jaw region), rotation math, Kabsch
alignment, the capture noise, dataset shapes, the least-squares baseline and that the solver learns.
