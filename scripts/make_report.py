"""Assemble all reproduction results into a Markdown report + merged CSV.

Reads results/summary/{aggregate.csv, all_runs.csv, posthoc_metrics.csv,
intervention_curves.csv} and writes:
  results/summary/REPORT.md          human-readable, paper-aligned tables
  results/summary/paper_tables.csv   one tidy row per (dataset, model) metric

Run collect_results.py and posthoc_metrics.py first.
"""
import csv
import os
from collections import defaultdict

import numpy as np

ROOT = "/research/hal-afsharim/minimal_cbm"
OUT = os.path.join(ROOT, "results", "summary")

DATASETS = ["mpi3d", "shapes3d", "cifar10", "cub12", "awa2"]
DS_LABEL = {"mpi3d": "MPI3D", "shapes3d": "Shapes3D", "cifar10": "CIFAR-10",
            "cub12": "CUB", "awa2": "AwA2"}
MODEL_ORDER = ["Vanilla", "CBM", "CEM", "ARCBM", "SCBM",
               "MCBM (low g)", "MCBM (medium g)", "MCBM (high g)"]


def read_csv(name):
    path = os.path.join(OUT, name)
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


def fnum(x):
    try:
        v = float(x)
        return v if v == v else None
    except (TypeError, ValueError):
        return None


def cell(agg, ds, model, mean_col, std_col=None):
    for r in agg:
        if r["dataset"] == ds and r["model"] == model:
            m = fnum(r.get(mean_col))
            if m is None:
                return "-"
            if std_col:
                s = fnum(r.get(std_col))
                return f"{m:.1f}±{s:.1f}" if s is not None else f"{m:.1f}"
            return f"{m:.1f}"
    return "-"


def table(agg, title, mean_col, std_col, datasets=DATASETS, note=""):
    lines = [f"### {title}", ""]
    if note:
        lines += [note, ""]
    header = "| Model | " + " | ".join(DS_LABEL[d] for d in datasets) + " |"
    sep = "|" + "---|" * (len(datasets) + 1)
    lines += [header, sep]
    for model in MODEL_ORDER:
        cells = [cell(agg, d, model, mean_col, std_col) for d in datasets]
        if all(c == "-" for c in cells):
            continue
        lines.append(f"| {model} | " + " | ".join(cells) + " |")
    lines.append("")
    return "\n".join(lines)


def posthoc_agg(posthoc):
    agg = defaultdict(lambda: defaultdict(list))
    for r in posthoc:
        key = (r["dataset"], r["model"])
        for m in ("cka", "disentanglement", "ois", "nis"):
            v = fnum(r.get(m))
            if v is not None:
                agg[key][m].append(v)
    out = []
    for (ds, model), md in agg.items():
        row = {"dataset": ds, "model": model}
        for m, vals in md.items():
            row[f"{m}_mean"] = np.mean(vals)
            row[f"{m}_std"] = np.std(vals)
        out.append(row)
    return out


def spearman(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    ok = ~(np.isnan(x) | np.isnan(y))
    x, y = x[ok], y[ok]
    if len(x) < 3:
        return None
    rx = np.argsort(np.argsort(x)); ry = np.argsort(np.argsort(y))
    rx = rx - rx.mean(); ry = ry - ry.mean()
    denom = np.sqrt((rx**2).sum() * (ry**2).sum())
    return float((rx * ry).sum() / denom) if denom else None


def main():
    agg = read_csv("aggregate.csv")
    posthoc = read_csv("posthoc_metrics.csv")
    ph = posthoc_agg(posthoc)

    md = ["# MCBM Reproduction Results",
          "",
          "Reproduction of *There Was Never a Bottleneck in Concept Bottleneck "
          "Models* (Almudévar et al., ICLR 2026, arXiv:2506.04877), with the "
          "additional concept-leakage metrics (OIS/NIS) and intervention "
          "AUC/NAUC from *Concepts' Information Bottleneck Models* (Galliamov "
          "et al., ICLR 2026, arXiv:2602.14626).",
          "",
          "All numbers are **mean ± std over seeds 42, 43, 44**. Nuisance "
          "leakage probes use a CUDA re-implementation of the paper's sklearn "
          "MLP probe (validated equivalent, see `scripts/probe_equivalence_test.py`).",
          "",
          "Models: Vanilla, CBM, CEM, ARCBM (repo `arhcbm`), SCBM (repo "
          "`shcbm`), and MCBM at three IB strengths (low/medium/high γ). ECBM "
          "is not implemented in the reference repository and is omitted.",
          ""]

    md.append(table(agg, "Table 6 — Task accuracy (%)",
                    "task_acc_mean", "task_acc_std"))
    md.append(table(agg, "Table 5 — Concept accuracy (%)",
                    "concept_acc_mean", "concept_acc_std",
                    datasets=["cifar10", "cub12", "awa2"],
                    note="_MPI3D/Shapes3D omitted: all models reach ~100%._"))
    md.append(table(agg, "Table 3 — Task-related information leakage, URR n_y (%)",
                    "urr_task_mean", "urr_task_std",
                    note="_Uncertainty Reduction Ratio; lower = less leakage._"))
    md.append(table(agg, "Table 4 — Task-unrelated information leakage, URR n_ȳ (%)",
                    "urr_nontask_mean", "urr_nontask_std",
                    datasets=["mpi3d", "shapes3d"],
                    note="_Only MPI3D and Shapes3D have task-unrelated nuisances._"))

    md += ["### Calibration & concept mAP", ""]
    md.append(table(agg, "Expected Calibration Error (ECE)",
                    "ece_mean", "ece_std"))
    md.append(table(agg, "Brier score", "brier_mean", "brier_std"))

    # Post-hoc interpretability (Fig. 4) + CIBM leakage
    md += ["### Figure 4 — Interpretability metrics & concept leakage", "",
           "_CKA (↑), Disentanglement (↑), OIS (↓), NIS (↓)._", ""]
    for metric, arrow in [("cka", "↑"), ("disentanglement", "↑"),
                          ("ois", "↓"), ("nis", "↓")]:
        md.append(table(ph, f"{metric.upper()} ({arrow})",
                        f"{metric}_mean", f"{metric}_std"))

    # Interventions (AUC/NAUC-TTI) — lowest-confidence policy (MCBM Fig. 5)
    md += ["### Interventions — lowest-confidence policy (MCBM Fig. 5)", ""]
    md.append(table(agg, "AUC-TTI (mean accuracy over intervention curve, %)",
                    "auc_tti_mean", "auc_tti_std"))
    md.append(table(agg, "NAUC-TTI (normalized accuracy gain per group)",
                    "nauc_tti_mean", "nauc_tti_std"))
    # Interventions — random policy (CIBM protocol, Table 3 / Fig. 3)
    md += ["### Interventions — random policy (CIBM protocol)", ""]
    md.append(table(agg, "AUC-TTI random (%)",
                    "auc_tti_random_mean", "auc_tti_random_std"))
    md.append(table(agg, "NAUC-TTI random",
                    "nauc_tti_random_mean", "nauc_tti_random_std"))

    # Table 7 — rank correlations between metric and URR across datasets
    md += ["### Table 7 — Rank correlation (Spearman) between each metric and "
           "URR n_y, across datasets", ""]
    # per-dataset mean URR (over models) vs per-dataset mean metric — the paper
    # correlates across MODELS within each dataset; we do the same.
    ph_by = {(r["dataset"], r["model"]): r for r in ph}
    agg_by = {(r["dataset"], r["model"]): r for r in agg}
    md += ["| Metric | " + " | ".join(DS_LABEL[d] for d in DATASETS) + " |",
           "|" + "---|" * (len(DATASETS) + 1)]
    for metric in ("cka", "disentanglement", "ois"):
        cells = []
        for d in DATASETS:
            xs, ys = [], []
            for model in MODEL_ORDER:
                pr = ph_by.get((d, model)); ar = agg_by.get((d, model))
                if pr and ar:
                    mv = fnum(pr.get(f"{metric}_mean"))
                    uv = fnum(ar.get("urr_task_mean"))
                    if mv is not None and uv is not None:
                        xs.append(mv); ys.append(uv)
            rho = spearman(xs, ys)
            cells.append(f"{rho:.2f}" if rho is not None else "-")
        md.append(f"| {metric.upper()} | " + " | ".join(cells) + " |")
    md.append("")

    md += ["### Notes", "",
           "- **URR** here is `mutual_inf_test/nuisances_* × 100`, i.e. the "
           "paper's Î(N_j;Z|C)/Ĥ(N_j) averaged over nuisances.",
           "- **AwA2** concepts = 20 non-degenerate appearance/morphology "
           "attributes (constant columns skipped); see "
           "`data/awa2/mcbm_awa2/manifest.json`.",
           "- **OIS/NIS** follow Espinosa Zarlenga et al. (2023) definitions; "
           "absolute scale may differ from a specific reference implementation, "
           "so compare models relative to each other.",
           "- Full per-seed numbers in `all_runs.csv`; per-config aggregates in "
           "`aggregate.csv`; intervention curves in `intervention_curves.csv`.",
           ""]

    with open(os.path.join(OUT, "REPORT.md"), "w") as f:
        f.write("\n".join(md))
    print("wrote", os.path.join(OUT, "REPORT.md"))


if __name__ == "__main__":
    main()
