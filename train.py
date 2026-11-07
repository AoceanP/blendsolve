"""Train the solver, compare it with the least-squares baseline and write results.json.

Usage: python train.py [--epochs 40] [--train 20000]
"""
import argparse
import json
import time

import torch

from blendsolve import data, model, rig


def evaluate(w_pred: torch.Tensor, w_true: torch.Tensor) -> dict:
    """Return weight MAE, jaw MAE and mean vertex error of the re-posed clean mesh."""
    base, d = rig.neutral(), rig.deltas(rig.neutral())
    v_err = (rig.pose(w_pred, base, d) - rig.pose(w_true, base, d)).norm(dim=-1).mean()
    return {"weight_mae": round((w_pred - w_true).abs().mean().item(), 4),
            "jaw_mae": round((w_pred[:, -1] - w_true[:, -1]).abs().mean().item(), 4),
            "vertex_error": round(v_err.item(), 5)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--train", type=int, default=20000)
    args = ap.parse_args()
    torch.manual_seed(0)

    x_tr, w_tr = data.make_dataset(args.train, seed=1)
    x_va, w_va = data.make_dataset(2000, seed=2)
    x_te, w_te = data.make_dataset(2000, seed=3)
    mu, sd = x_tr.mean(0), x_tr.std(0) + 1e-6

    net = model.Solver(x_tr.shape[1], w_tr.shape[1])
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    best, best_state = float("inf"), None
    t0 = time.time()
    for epoch in range(args.epochs):
        net.train()
        for idx in torch.randperm(len(x_tr)).split(256):
            loss = torch.nn.functional.l1_loss(net((x_tr[idx] - mu) / sd), w_tr[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()
        net.eval()
        with torch.no_grad():
            val = torch.nn.functional.l1_loss(net((x_va - mu) / sd), w_va).item()
        if val < best:                                                  # keep the best checkpoint
            best, best_state = val, {k: v.clone() for k, v in net.state_dict().items()}
        if epoch % 5 == 0 or epoch == args.epochs - 1:
            print(f"epoch {epoch:3d}  train {loss.item():.4f}  val {val:.4f}")
    train_s = time.time() - t0
    net.load_state_dict(best_state)

    with torch.no_grad():
        nn_res = evaluate(net((x_te - mu) / sd), w_te)
        ls_res = evaluate(model.least_squares(x_te), w_te)
        frame = (x_te[:1] - mu) / sd
        for _ in range(20):
            net(frame)
        t1 = time.perf_counter()
        for _ in range(500):
            net(frame)
        ms = (time.perf_counter() - t1) / 500 * 1000

    results = {"neural": nn_res, "least_squares": ls_res, "train_seconds": round(train_s, 1),
               "inference_ms_per_frame": round(ms, 3), "params": sum(p.numel() for p in net.parameters()),
               "train_samples": args.train, "test_samples": len(x_te)}
    print(json.dumps(results, indent=2))
    json.dump(results, open("results.json", "w"), indent=2)
    torch.save({"state": net.state_dict(), "mu": mu, "sd": sd}, "solver.pt")


if __name__ == "__main__":
    main()
