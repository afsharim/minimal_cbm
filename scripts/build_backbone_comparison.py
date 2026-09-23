"""Backbone-matched MCBM vs CIBM comparison (full metric set).

MCBM: results/summary/aggregate.csv (+ posthoc_metrics.csv for CKA/DCI/OIS/NIS).
CIBM (backbone-matched, MCBM's own CBM encoder features):
      ../cibm/outputs/summary/cibm_fullmetrics_mcbmenc.csv
Writes results/summary/MCBM_vs_CIBM_backbone_matched.md.
Datasets with results now: mpi3d, shapes3d, cifar10, awa2 (CUB re-running).
"""
import csv
import collections
import os

import numpy as np

ROOT = "/research/hal-afsharim/minimal_cbm"
CIBM = "/research/hal-afsharim/cibm/outputs/summary/cibm_fullmetrics_mcbmenc.csv"
OUT = f"{ROOT}/results/summary/MCBM_vs_CIBM_backbone_matched.md"
DS = ["mpi3d", "shapes3d", "cifar10", "awa2"]
DSL = {"mpi3d": "MPI3D", "shapes3d": "Shapes3D", "cifar10": "CIFAR-10", "awa2": "AwA2"}
# metric key in this script -> (MCBM aggregate col, CIBM fullmetrics col)
MET = [("Task Acc", "task_acc", "task_acc"),
       ("Concept Acc", "concept_acc", "concept_acc"),
       ("Task leak(pp)", "leak_acc_task", "task_leak"),
       ("Nontask leak", "leak_acc_nontask", "nontask_leak"),
       ("URR task", "urr_task", "urr_task"),
       ("URR nontask", "urr_nontask", "urr_nontask"),
       ("ECE", "ece", "ece"), ("Brier", "brier", "brier"),
       ("mAP", "map_concepts", "map"),
       ("CKA", "cka", "cka"), ("DCI", "disentanglement", "disentanglement"),
       ("OIS", "ois", "ois"), ("NIS", "nis", "nis")]


def load_mcbm():
    agg = {}
    for r in csv.DictReader(open(f"{ROOT}/results/summary/aggregate.csv")):
        agg[(r["dataset"], r["model"])] = r
    ph = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in csv.DictReader(open(f"{ROOT}/results/summary/posthoc_metrics.csv")):
        for m in ("cka", "disentanglement", "ois", "nis"):
            v = r.get(m)
            if v not in (None, "", "None", "nan"):
                ph[(r["dataset"], r["model"])][m].append(float(v))
    return agg, ph


def load_cibm():
    g = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in csv.DictReader(open(CIBM)):
        for _, _, c in MET:
            v = r.get(c)
            if v not in (None, "", "None", "nan"):
                g[(r["dataset"], r["variant"])][c].append(float(v))
    return g


def mval(agg, ph, ds, model, mcol):
    if mcol in ("cka", "disentanglement", "ois", "nis"):
        vs = ph[(ds, model)].get(mcol, [])
        return np.mean(vs) if vs else None
    r = agg.get((ds, model))
    if not r:
        return None
    v = r.get(f"{mcol}_mean")
    return float(v) if v not in (None, "") else None


def cval(g, ds, variant, ccol):
    vs = g[(ds, variant)].get(ccol, [])
    return np.mean(vs) if vs else None


def fmt(v, dec=2):
    return "--" if v is None else f"{v:.{dec}f}"


def main():
    agg, ph = load_mcbm()
    cib = load_cibm()
    L = ["# MCBM vs CIBM — backbone-matched, full metric suite\n",
         "CIBM's IB head trained on **MCBM's own CBM-encoder features** (frozen), "
         "so both methods use the **same backbone, same concepts, same task** — "
         "the difference is method only (MCBM co-trains encoder+head; CIBM freezes "
         "the encoder and trains the head). Seeds 42/43/44. All metrics via MCBM's "
         "identical code (CUDA leakage probe, posthoc CKA/DCI/OIS/NIS).\n",
         "> **CUB is being re-run** (instance-level fix + backbone extraction) and "
         "will be added when it completes.\n"]
    pairs = [("MCBM · CBM", "CBM", "MCBM"), ("CIBM · Basic", "basic", "CIBM"),
             ("MCBM · mcbm γ-low", "MCBM (low g)", "MCBM"),
             ("CIBM · IB-var", "ibvar", "CIBM"),
             ("MCBM · mcbm γ-med", "MCBM (medium g)", "MCBM"),
             ("MCBM · mcbm γ-high", "MCBM (high g)", "MCBM"),
             ("CIBM · IB-entropy", "ibhc", "CIBM")]
    hdr = "| Method | " + " | ".join(m[0] for m in MET) + " |"
    sep = "|" + "---|" * (len(MET) + 1)
    for ds in DS:
        L.append(f"## {DSL[ds]}\n")
        L.append(hdr); L.append(sep)
        for label, key, src in pairs:
            cells = []
            for _, mcol, ccol in MET:
                if src == "MCBM":
                    v = mval(agg, ph, ds, key, mcol)
                else:
                    v = cval(cib, ds, key, ccol)
                dec = 4 if mcol in ("ece", "brier") else 2
                cells.append(fmt(v, dec))
            L.append(f"| {label} | " + " | ".join(cells) + " |")
        L.append("")
    with open(OUT, "w") as f:
        f.write("\n".join(L) + "\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
