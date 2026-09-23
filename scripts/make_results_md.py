"""Write a human-readable Markdown report of all completed runs.

Reads results/summary/all_runs.csv (produced by collect_results.py, complete
runs only) and emits results/summary/RESULTS.md with, per dataset:
  1. a mean +/- std table across seeds (one row per model config)  <- main ask
  2. a per-seed detail table (one row per individual run)

Metrics reported (paper 2506.04877 + CIBM 2602.14626):
  Task Acc, Concept Acc, URR (task / non-task nuisances), ECE, Brier,
  mAP(concepts), and intervention AUC-TTI / NAUC-TTI where applicable.
"""
import csv
import os
from collections import defaultdict

import numpy as np

ROOT = "/research/hal-afsharim/minimal_cbm"
SUMMARY = os.path.join(ROOT, "results", "summary")
ALL_RUNS = os.path.join(SUMMARY, "all_runs.csv")
OUT = os.path.join(SUMMARY, "RESULTS.md")

# Expected coverage: 8 configs x 3 seeds per dataset.
DATASET_ORDER = ["mpi3d", "shapes3d", "cifar10", "cub12", "awa2"]
DATASET_TITLE = {"mpi3d": "MPI3D", "shapes3d": "Shapes3D", "cifar10": "CIFAR-10",
                 "cub12": "CUB", "awa2": "AwA2"}
SEEDS = [42, 43, 44]
CONFIGS_PER_DATASET = 8

# (csv column, header, n decimals)
METRICS = [
    ("task_acc",         "Task Acc",           2),
    ("concept_acc",      "Concept Acc",        2),
    ("leak_acc_task",    "Task leak (pp)",     2),
    ("leak_acc_nontask", "Non-task leak (pp)", 2),
    ("urr_task",         "URR task",           2),
    ("urr_nontask",      "URR non-task",       2),
    ("ece",              "ECE",                4),
    ("brier",            "Brier",              4),
    ("map_concepts",     "mAP",                4),
    ("cka",              "CKA",                3),
    ("disentanglement",  "Disentangle",        3),
    ("ois",              "OIS",                2),
    ("nis",              "NIS",                2),
    ("auc_tti",          "AUC-TTI",            2),
    ("nauc_tti",         "NAUC-TTI",           4),
]
# model display order within a dataset
MODEL_ORDER = ["Vanilla", "CBM", "CEM", "ARCBM", "SCBM",
               "MCBM (low g)", "MCBM (medium g)", "MCBM (high g)"]


def fnum(x):
    if x is None or x == "":
        return None
    try:
        v = float(x)
    except ValueError:
        return None
    if v != v:  # NaN -> treat as missing
        return None
    return v


def load():
    rows = []
    with open(ALL_RUNS) as f:
        for r in csv.DictReader(f):
            rows.append(r)
    # merge post-hoc metrics (CKA / disentanglement / OIS / NIS) if present
    ph_path = os.path.join(SUMMARY, "posthoc_metrics.csv")
    ph = {}
    if os.path.exists(ph_path):
        with open(ph_path) as f:
            for r in csv.DictReader(f):
                ph[(r["config"], str(r["seed"]))] = r
    for r in rows:
        p = ph.get((r["config"], str(r["seed"])))
        if p:
            for k in ("cka", "disentanglement", "ois", "nis"):
                r[k] = p.get(k)
    return rows


def fmt_mean_std(vals, dec):
    vals = [v for v in vals if v is not None]
    if not vals:
        return "--"
    m = np.mean(vals)
    if len(vals) > 1:
        s = np.std(vals)
        return f"{m:.{dec}f} ± {s:.{dec}f}"
    return f"{m:.{dec}f}"


def fmt_one(v, dec):
    return "--" if v is None else f"{v:.{dec}f}"


def model_sort_key(model):
    return MODEL_ORDER.index(model) if model in MODEL_ORDER else 99


def main():
    rows = load()
    # group by dataset -> config -> list of seed-rows
    by_ds = defaultdict(lambda: defaultdict(list))
    for r in rows:
        by_ds[r["dataset"]][r["config"]].append(r)

    present_ds = [d for d in DATASET_ORDER if d in by_ds] + \
                 [d for d in by_ds if d not in DATASET_ORDER]

    lines = []
    lines.append("# MCBM Reproduction — Results\n")
    lines.append("Reproduction of *There Was Never a Bottleneck in Concept "
                 "Bottleneck Models* (arXiv:2506.04877), with the leakage / "
                 "impurity metrics of *Concepts' Information Bottleneck Models* "
                 "(arXiv:2602.14626). Seeds: 42, 43, 44.\n")

    # ---- coverage summary ----
    total_done = len(rows)
    lines.append("## Coverage\n")
    lines.append(f"**{total_done} / 120** runs complete "
                 "(5 datasets × 8 model configs × 3 seeds). "
                 "This report is regenerated as runs finish; incomplete "
                 "(dataset, config) cells show the seeds available so far.\n")
    lines.append("| Dataset | Configs complete | Runs complete | Seeds |")
    lines.append("|---|---|---|---|")
    for ds in present_ds:
        cfgs = by_ds[ds]
        nruns = sum(len(v) for v in cfgs.values())
        full_cfgs = sum(1 for v in cfgs.values() if len(v) == len(SEEDS))
        seeds_seen = sorted({int(r["seed"]) for v in cfgs.values() for r in v})
        lines.append(f"| {DATASET_TITLE.get(ds, ds)} | "
                     f"{full_cfgs}/{CONFIGS_PER_DATASET} full ({nruns}/{CONFIGS_PER_DATASET*len(SEEDS)} runs) | "
                     f"{nruns} | {','.join(map(str, seeds_seen))} |")
    lines.append("")

    lines.append(
        "**Metrics.** Task/Concept Acc in %. "
        "**Task leak (pp)** / **Non-task leak (pp)** = accuracy gain (percentage "
        "points) for predicting a task-related / task-unrelated nuisance from "
        "the joint [z, c] versus from concepts c alone — the paper's leakage "
        "measure; lower = less leakage. **URR task / non-task** = the "
        "mutual-information version of the same (×100), lower = better. "
        "ECE/Brier = calibration (lower better). mAP = mean average precision of "
        "concept prediction. **CKA** = linear centered-kernel alignment of the "
        "representation z with the concepts (Fig. 4, higher = better). "
        "**Disentangle** = DCI disentanglement (Fig. 4, higher = better). "
        "**OIS** = Oracle Impurity Score and **NIS** = Niche Impurity Score "
        "(CIBM, arXiv:2602.14626; ×100, lower = purer concepts). "
        "**AUC-TTI / NAUC-TTI** = test-time-intervention curve area "
        "(lowest-confidence policy), intervenable models only. "
        "'--' = not applicable (e.g. no concept head) or not yet computed.\n")

    hdr = "| Model | Seeds | " + " | ".join(h for _, h, _ in METRICS) + " |"
    sep = "|" + "---|" * (len(METRICS) + 2)

    for ds in present_ds:
        cfgs = by_ds[ds]
        lines.append(f"## {DATASET_TITLE.get(ds, ds)}\n")

        # order configs by model order
        def cfg_key(cfg):
            model = cfgs[cfg][0]["model"]
            return model_sort_key(model)
        ordered = sorted(cfgs, key=cfg_key)

        # ---- mean +/- std table ----
        lines.append("### Mean ± std across seeds\n")
        lines.append(hdr)
        lines.append(sep)
        for cfg in ordered:
            rs = cfgs[cfg]
            model = rs[0]["model"]
            cells = []
            for col, _, dec in METRICS:
                cells.append(fmt_mean_std([fnum(r.get(col)) for r in rs], dec))
            lines.append(f"| {model} | {len(rs)} | " + " | ".join(cells) + " |")
        lines.append("")

        # ---- per-seed detail ----
        lines.append("### Per-seed detail\n")
        lines.append("| Model | Seed | " + " | ".join(h for _, h, _ in METRICS) + " |")
        lines.append(sep)
        for cfg in ordered:
            for r in sorted(cfgs[cfg], key=lambda x: int(x["seed"])):
                model = r["model"]
                cells = [fmt_one(fnum(r.get(col)), dec) for col, _, dec in METRICS]
                lines.append(f"| {model} | {r['seed']} | " + " | ".join(cells) + " |")
        lines.append("")

    with open(OUT, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote {OUT} ({total_done} runs, {len(present_ds)} datasets)")


if __name__ == "__main__":
    main()
