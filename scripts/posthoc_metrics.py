"""Post-hoc interpretability metrics from saved predictions.

For each finished run (results/<config>/<seed>/predictions/epoch_<E>.pth and
epoch_<E>_train.pth) computes:
  - CKA(z, concepts)     : linear Centered Kernel Alignment, higher = better
                           (MCBM paper Fig. 4 left).
  - DCI disentanglement  : Eastwood & Williams (2018), higher = better
                           (MCBM paper Fig. 4 middle).
  - OIS  (Oracle Impurity Score, lower = better): concept leakage measured as
                           the deviation of the soft-concept purity matrix from
                           the ground-truth (oracle) purity matrix, following
                           Espinosa Zarlenga et al. (2023). Reported x100.
  - NIS  (Niche Impurity Score, lower = better): mean extra AUC for predicting
                           each true concept from the FULL soft-concept vector
                           beyond its own dimension (distributed leakage proxy,
                           following the same reference). Reported x100.

Writes results/summary/posthoc_metrics.csv.

Usage:
    python scripts/posthoc_metrics.py [dataset ...] [--force]
  - optional dataset filters (e.g. `mpi3d shapes3d`) restrict which runs to do.
  - incremental by default: (config, seed) already in the CSV are skipped unless
    --force. Parallel across runs (MCBM_POSTHOC_WORKERS, default 8).

Notes on faithful reproduction:
  * CKA uses a subsample (<= CKA_N points) because the kernel is O(n^2).
  * DCI/OIS/NIS use a subsample (<= PROBE_N points) for tractable probe fits.
  * A fresh RandomState(0) is created per run, so results are identical whether
    the pass runs sequentially or in parallel.
  * OIS/NIS follow the cited definitions; absolute scale may differ slightly
    from a given reference implementation, so use them for relative comparison.
"""
import csv
import glob
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import torch
import yaml

sys.path.insert(0, "/research/hal-afsharim/minimal_cbm")
from src.helpers.alignment import AlignmentMetrics
from src.helpers.disentanglement.dci import dci
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression

ROOT = "/research/hal-afsharim/minimal_cbm"
RESULTS = os.path.join(ROOT, "results")
OUT = os.path.join(RESULTS, "summary")
CSV_PATH = os.path.join(OUT, "posthoc_metrics.csv")
CKA_N = 4096
PROBE_N = 8000
WORKERS = int(os.environ.get("MCBM_POSTHOC_WORKERS", "8"))

MODEL_LABEL = {"vanilla": "Vanilla", "cbm": "CBM", "cem": "CEM",
               "arhcbm": "ARCBM", "shcbm": "SCBM"}
GAMMA_ORDER = {"mpi3d": [1.0, 3.0, 5.0], "shapes3d": [1.0, 3.0, 5.0],
               "cifar10": [0.1, 0.3, 0.5], "cub12": [0.05, 0.1, 0.3],
               "awa2": [0.05, 0.1, 0.3]}
COLS = ["config", "dataset", "model", "seed", "cka", "disentanglement",
        "ois", "nis"]


_NE = {}


def n_epochs_of(config):
    if config not in _NE:
        sub = config.split("-")[0]
        try:
            with open(os.path.join(ROOT, "configs", sub, config + ".yaml")) as f:
                _NE[config] = yaml.safe_load(f)["training"]["n_epochs"]
        except Exception:
            _NE[config] = None
    return _NE[config]


def is_complete(config, seed):
    """True only when the run reached its final epoch. Post-hoc must use FINAL
    predictions and must not cache metrics computed from an intermediate
    checkpoint (save_epochs < n_epochs), which would go stale after the run
    finishes."""
    fm = os.path.join(RESULTS, config, str(seed), "final_metrics.json")
    if not os.path.exists(fm):
        return False
    try:
        e = json.load(open(fm)).get("epoch", 0)
    except Exception:
        return False
    ne = n_epochs_of(config)
    return ne is not None and e >= ne


def parse_config_name(config):
    ds, rest = config.split("-", 1)
    if rest.startswith("mcbm"):
        g = rest.split("-", 1)[1]
        gamma = {"005": 0.05, "01": 0.1, "03": 0.3, "05": 0.5,
                 "1": 1.0, "3": 3.0, "5": 5.0}.get(g, g)
        level = None
        if ds in GAMMA_ORDER and gamma in GAMMA_ORDER[ds]:
            level = ["low", "medium", "high"][GAMMA_ORDER[ds].index(gamma)]
        return ds, f"MCBM ({level} g)" if level else f"MCBM (g={gamma})"
    return ds, MODEL_LABEL.get(rest, rest)


def subsample(rng, n, cap):
    if n <= cap:
        return np.arange(n)
    return rng.choice(n, cap, replace=False)


def compute_cka(z, c, rng):
    idx = subsample(rng, z.shape[0], CKA_N)
    za = torch.tensor(z[idx], dtype=torch.float32)
    cb = torch.tensor(c[idx], dtype=torch.float32)
    za = za - za.mean(0)
    cb = cb - cb.mean(0)
    try:
        return float(AlignmentMetrics.cka(za, cb, kernel_metric='ip'))
    except Exception:
        return float('nan')


def compute_dci(z, c, rng):
    idx = subsample(rng, z.shape[0], PROBE_N)
    factors = c[idx].astype(np.float64)
    codes = z[idx].astype(np.float64)
    keep = [j for j in range(factors.shape[1]) if factors[:, j].std() > 1e-8]
    if len(keep) < 2:
        return float('nan')
    try:
        disent, _, _ = dci(factors[:, keep], codes,
                           continuous_factors=True, model='lasso')
        return float(disent)
    except Exception:
        return float('nan')


def _auc(feat, target):
    if len(np.unique(target)) < 2:
        return 0.5
    a = roc_auc_score(target, feat)
    return max(a, 1 - a)


def compute_ois(c_soft, c_true, rng):
    idx = subsample(rng, c_soft.shape[0], PROBE_N)
    cs, ct = c_soft[idx], c_true[idx]
    k = ct.shape[1]
    keep = [j for j in range(k) if len(np.unique(ct[:, j])) > 1]
    if len(keep) < 2:
        return float('nan')
    cs, ct = cs[:, keep], ct[:, keep]
    k = len(keep)
    P_soft = np.zeros((k, k))
    P_oracle = np.zeros((k, k))
    for i in range(k):
        for j in range(k):
            P_soft[i, j] = _auc(cs[:, i], ct[:, j])
            P_oracle[i, j] = 1.0 if i == j else _auc(ct[:, i], ct[:, j])
    P_soft = 2 * (P_soft - 0.5)
    P_oracle = 2 * (P_oracle - 0.5)
    ois = np.sqrt(((P_soft - P_oracle) ** 2).sum()) / k
    return float(ois * 100)


def compute_nis(c_soft, c_true, rng):
    idx = subsample(rng, c_soft.shape[0], PROBE_N)
    cs, ct = c_soft[idx], c_true[idx]
    k = ct.shape[1]
    keep = [j for j in range(k) if len(np.unique(ct[:, j])) > 1]
    if len(keep) < 2:
        return float('nan')
    cs, ct = cs[:, keep], ct[:, keep]
    k = len(keep)
    extras = []
    for j in range(k):
        own = _auc(cs[:, j], ct[:, j])
        try:
            lr = LogisticRegression(max_iter=200, C=1.0)
            lr.fit(cs, ct[:, j])
            full = roc_auc_score(ct[:, j], lr.predict_proba(cs)[:, 1])
            full = max(full, 1 - full)
        except Exception:
            full = own
        extras.append(max(0.0, full - own))
    return float(np.mean(extras) * 100)


def load_preds(rdir):
    test_files = sorted(glob.glob(os.path.join(rdir, "predictions", "epoch_*.pth")))
    test_files = [f for f in test_files if not f.endswith("_train.pth")]
    if not test_files:
        return None
    # highest epoch number, not lexicographic (epoch_100 > epoch_50)
    test_files.sort(key=lambda f: int(os.path.basename(f).split("_")[1].split(".")[0]))
    te = torch.load(test_files[-1], map_location="cpu")
    if "z" not in te or "c" not in te:
        return None
    z = te["z"].float().numpy()
    c = te["c"].float().numpy()
    # Vanilla has no concept head -> no c_preds. CKA/DCI(z, c) are still valid;
    # OIS/NIS (concept-side) are left NaN in that case.
    if "c_preds" in te:
        c_soft = te["c_preds"].float().numpy()
        if c_soft.ndim == 3:
            c_soft = c_soft[:, :, 0]
    else:
        c_soft = None
    return z, c, c_soft


def process_run(config, seed):
    """Worker: compute all post-hoc metrics for one run. Deterministic."""
    rdir = os.path.join(RESULTS, config, str(seed))
    preds = load_preds(rdir)
    if preds is None:
        return None
    z, c, c_soft = preds
    ds, model = parse_config_name(config)
    rng = np.random.RandomState(0)   # fresh per run -> parallel-invariant
    row = {"config": config, "dataset": ds, "model": model, "seed": int(seed)}
    row["cka"] = compute_cka(z, c, rng)
    row["disentanglement"] = compute_dci(z, c, rng)
    if c_soft is not None and c_soft.shape == c.shape and c_soft.std() > 0:
        row["ois"] = compute_ois(c_soft, c, rng)
        row["nis"] = compute_nis(c_soft, c, rng)
    else:
        row["ois"] = float('nan')
        row["nis"] = float('nan')
    return row


def load_existing():
    if not os.path.exists(CSV_PATH):
        return {}
    out = {}
    with open(CSV_PATH) as f:
        for r in csv.DictReader(f):
            out[(r["config"], int(r["seed"]))] = r
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    argv = [a for a in sys.argv[1:]]
    force = "--force" in argv
    ds_filter = [a for a in argv if not a.startswith("--")]

    existing = {} if force else load_existing()
    # Drop any cached rows for runs that are no longer complete (e.g. metrics
    # computed earlier from an intermediate checkpoint): they will be recomputed
    # from FINAL predictions once the run finishes.
    existing = {k: v for k, v in existing.items() if is_complete(k[0], k[1])}

    todo = []
    for config in sorted(os.listdir(RESULTS)):
        cdir = os.path.join(RESULTS, config)
        if not os.path.isdir(cdir) or config in ("logs", "summary") or "smoke" in config:
            continue
        if ds_filter and config.split("-")[0] not in ds_filter:
            continue
        for seed in sorted(os.listdir(cdir)):
            if not os.path.isdir(os.path.join(cdir, seed)):
                continue
            if (config, int(seed)) in existing:
                continue
            # only FINAL predictions of COMPLETE runs (no stale intermediates)
            if not is_complete(config, int(seed)):
                continue
            if load_preds(os.path.join(cdir, seed)) is None:
                continue
            todo.append((config, int(seed)))

    print(f"{len(existing)} cached, {len(todo)} to compute "
          f"({WORKERS} workers)")
    rows = list(existing.values())
    if todo:
        with ProcessPoolExecutor(max_workers=WORKERS) as ex:
            futs = {ex.submit(process_run, c, s): (c, s) for c, s in todo}
            for fut in as_completed(futs):
                r = fut.result()
                if r is None:
                    continue
                rows.append(r)
                print(f"{r['config']} s{r['seed']}: CKA={r['cka']:.3f} "
                      f"Disent={r['disentanglement']:.3f} "
                      f"OIS={r['ois']:.3f} NIS={r['nis']:.3f}", flush=True)

    # stable order: dataset, config, seed
    def sort_key(r):
        return (str(r["dataset"]), str(r["config"]), int(r["seed"]))
    rows.sort(key=sort_key)
    with open(CSV_PATH, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in COLS})
    print(f"\n{len(rows)} runs -> {CSV_PATH}")


if __name__ == "__main__":
    main()
