"""Validate that the CUDA probe matches the sklearn MLPClassifier probe.

Compares get_results_classifier_torch vs get_results_classifier_sklearn on
synthetic data shaped like the real probe workloads:
  - MPI3D-like:  z(14)+c(14) features, 40-class nuisance, 130k train
  - CIFAR-like:  z(64)+c(64) features, binary nuisance, 50k train
  - CUB-like:    z(~50)+c(~50) features, binary nuisance, 4.8k train
Reports accuracy/mi from both backends and timing. Pass criterion: the
backends agree within the probe's own seed-to-seed noise (measured by
running the torch probe with 3 seeds).
"""
import sys
import time

import numpy as np
import torch

sys.path.insert(0, "/research/hal-afsharim/minimal_cbm")
from src.helpers.metrics import (
    get_results_classifier_sklearn, get_results_classifier_torch)


def make_data(n_train, n_test, d, n_classes, seed, signal=2.0):
    rng = np.random.RandomState(seed)
    W = rng.randn(d, n_classes) * signal / np.sqrt(d)
    def gen(n):
        x = rng.randn(n, d).astype(np.float32)
        logits = x @ W + rng.randn(n, n_classes) * 1.0
        y = logits.argmax(1).astype(np.int64)
        return torch.tensor(x), torch.tensor(y)
    xtr, ytr = gen(n_train)
    xte, yte = gen(n_test)
    return xtr, ytr, xte, yte


CASES = [
    ("mpi3d-like", 130_000, 43_000, 28, 40),
    ("cifar-like", 50_000, 10_000, 128, 2),
    ("cub-like", 4_800, 5_800, 100, 2),
]

for name, n_tr, n_te, d, k in CASES:
    xtr, ytr, xte, yte = make_data(n_tr, n_te, d, k, seed=0)
    t0 = time.time()
    res_sk = get_results_classifier_sklearn(xtr, ytr, xte, yte)
    t_sk = time.time() - t0

    torch_res = []
    t0 = time.time()
    for s in (42, 43, 44):
        torch_res.append(get_results_classifier_torch(
            xtr, ytr, xte, yte, random_state=s, device='cuda'))
    t_t = (time.time() - t0) / 3

    accs = [r['accuracy'] for r in torch_res]
    mis = [r['mi'] for r in torch_res]
    print(f"[{name}] sklearn: acc={res_sk['accuracy']:.3f} mi={res_sk['mi']:.4f} "
          f"({t_sk:.1f}s)")
    print(f"[{name}] torch  : acc={np.mean(accs):.3f}±{np.std(accs):.3f} "
          f"mi={np.mean(mis):.4f}±{np.std(mis):.4f} ({t_t:.1f}s/fit, "
          f"speedup x{t_sk/t_t:.1f})")
    d_acc = abs(res_sk['accuracy'] - np.mean(accs))
    d_mi = abs(res_sk['mi'] - np.mean(mis))
    tol_acc = max(0.5, 4 * np.std(accs))
    tol_mi = max(0.01, 4 * np.std(mis))
    status = "PASS" if (d_acc <= tol_acc and d_mi <= tol_mi) else "FAIL"
    print(f"[{name}] delta acc={d_acc:.3f} (tol {tol_acc:.3f}) "
          f"delta mi={d_mi:.4f} (tol {tol_mi:.4f}) -> {status}\n")
