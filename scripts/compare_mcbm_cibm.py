"""Side-by-side comparison: our MCBM reproduction vs CIBM (dsb-ifi/cibm), on the
same datasets. Reads:
  results/summary/aggregate.csv                (MCBM, our runs)
  ../cibm/outputs/summary/cibm_all_runs.csv    (CIBM, per-run -> aggregated here)
Writes results/summary/MCBM_vs_CIBM.md.

Compared metrics are the ones both pipelines produce: Task Acc, Concept Acc,
AUC-TTI, NAUC-TTI. IMPORTANT CAVEAT surfaced in the doc: MCBM trains the encoder
end-to-end; CIBM trains an MLP head on FROZEN backbone embeddings, so backbone-
level capacity differs (largest effect on the synthetic datasets).
"""
import csv
import os
from collections import defaultdict

import numpy as np

ROOT = "/research/hal-afsharim/minimal_cbm"
MCBM_AGG = f"{ROOT}/results/summary/aggregate.csv"
CIBM_CSV = "/research/hal-afsharim/cibm/outputs/summary/cibm_all_runs.csv"
OUT = f"{ROOT}/results/summary/MCBM_vs_CIBM.md"

DS_ORDER = ["mpi3d", "shapes3d", "cifar10", "cub", "awa2"]
DS_LABEL = {"mpi3d": "MPI3D", "shapes3d": "Shapes3D", "cifar10": "CIFAR-10",
            "cub": "CUB", "awa2": "AwA2"}
MCBM_DS = {"cub": "cub12"}  # cibm-name -> mcbm-name
CIBM_VAR_LABEL = {"basic": "Basic CBM", "ibvar": "IB-CBM (variational)",
                  "ibhc": "IB-CBM (entropy)"}


def load_mcbm():
    out = {}
    for r in csv.DictReader(open(MCBM_AGG)):
        out[(r["dataset"], r["model"])] = r
    return out


def load_cibm():
    g = defaultdict(lambda: defaultdict(list))
    for r in csv.DictReader(open(CIBM_CSV)):
        for m in ("task_acc", "concept_acc", "task_leak", "nontask_leak",
                  "urr_task", "urr_nontask", "auc_tti", "nauc_tti"):
            v = r.get(m)
            if v not in (None, "", "None"):
                g[(r["dataset"], r["variant"])][m].append(float(v))
    return g


def ms(vals, dec=2):
    vals = [v for v in vals if v is not None]
    if not vals:
        return "--"
    return f"{np.mean(vals):.{dec}f} ± {np.std(vals):.{dec}f}" if len(vals) > 1 \
        else f"{np.mean(vals):.{dec}f}"


def mcbm_cell(row, key, dec=2):
    if row is None:
        return "--"
    m, s = row.get(f"{key}_mean"), row.get(f"{key}_std")
    if m in (None, ""):
        return "--"
    return f"{float(m):.{dec}f} ± {float(s):.{dec}f}" if s not in (None, "") \
        else f"{float(m):.{dec}f}"


def row_line(method, task, con, tleak, ntleak, urr, auc, nauc):
    return f"| {method} | {task} | {con} | {tleak} | {ntleak} | {urr} | {auc} | {nauc} |"


def main():
    mcbm, cibm = load_mcbm(), load_cibm()
    L = []
    L.append("# MCBM vs CIBM — side-by-side on the same datasets\n")
    L.append("Our reproduction of *There Was Never a Bottleneck in CBMs* "
             "(arXiv:2506.04877; **MCBM**) versus *Concepts' Information "
             "Bottleneck Models* (arXiv:2602.14626, dsb-ifi/cibm; **CIBM**), run "
             "on the same datasets, seeds 42/43/44. Metrics both pipelines "
             "produce: Task Acc, Concept Acc (%), AUC-TTI (mean intervention-curve "
             "accuracy, %), NAUC-TTI (normalized intervention AUC).\n")
    L.append("> **Read this first — the setups are not backbone-matched.** MCBM "
             "trains the image encoder **end-to-end**; CIBM trains an MLP head on "
             "**frozen** backbone embeddings (InceptionV3 for CUB, xlsa17 "
             "ResNet-101 for AwA2, ImageNet ResNet-50 for CIFAR/MPI3D/Shapes3D). "
             "On the synthetic datasets a trained encoder reaches ~100% task acc "
             "while frozen ImageNet features cannot, so **task-accuracy gaps on "
             "MPI3D/Shapes3D are dominated by the backbone difference, not the "
             "method.** The informative within-method signal is how each paper's "
             "IB regularizer moves the numbers relative to its own baseline CBM.\n")
    L.append("> **Opposing IB behaviour (the key finding).** CIBM's IB regularizer "
             "*raises* task accuracy over its Basic CBM (e.g. Shapes3D 78.9→99.9, "
             "MPI3D 73.5→87.8); MCBM's γ-bottleneck *lowers* it as γ grows (MPI3D "
             "CBM 100 → mcbm-γ 65→25). The two 'information bottleneck' recipes "
             "push task accuracy in opposite directions.\n")

    for ds in DS_ORDER:
        mds = MCBM_DS.get(ds, ds)
        L.append(f"## {DS_LABEL[ds]}\n")
        L.append("| Method | Task Acc | Concept Acc | Task leak (pp) | "
                 "Non-task leak (pp) | URR task | AUC-TTI | NAUC-TTI |")
        L.append("|---|---|---|---|---|---|---|---|")
        # --- baseline CBM ---
        cb = mcbm.get((mds, "CBM"))
        L.append(row_line("MCBM · CBM (trained enc.)",
                          mcbm_cell(cb, "task_acc"), mcbm_cell(cb, "concept_acc"),
                          mcbm_cell(cb, "leak_acc_task"), mcbm_cell(cb, "leak_acc_nontask"),
                          mcbm_cell(cb, "urr_task"),
                          mcbm_cell(cb, "auc_tti"), mcbm_cell(cb, "nauc_tti", 4)))
        cg = cibm.get((ds, "basic"))
        if cg:
            L.append(row_line("CIBM · Basic CBM (frozen emb.)",
                              ms(cg["task_acc"]), ms(cg["concept_acc"]),
                              ms(cg["task_leak"]), ms(cg["nontask_leak"]),
                              ms(cg["urr_task"]),
                              ms(cg["auc_tti"]), ms(cg["nauc_tti"], 4)))
        # --- IB-regularized ---
        for m in ("MCBM (low g)", "MCBM (medium g)", "MCBM (high g)"):
            r = mcbm.get((mds, m))
            if r:
                lbl = "MCBM · " + m.replace("MCBM ", "").replace("(", "γ ").replace(")", "")
                L.append(row_line(lbl,
                                  mcbm_cell(r, "task_acc"), mcbm_cell(r, "concept_acc"),
                                  mcbm_cell(r, "leak_acc_task"), mcbm_cell(r, "leak_acc_nontask"),
                                  mcbm_cell(r, "urr_task"),
                                  mcbm_cell(r, "auc_tti"), mcbm_cell(r, "nauc_tti", 4)))
        for v in ("ibvar", "ibhc"):
            cg = cibm.get((ds, v))
            if cg:
                L.append(row_line(f"CIBM · {CIBM_VAR_LABEL[v]} (frozen emb.)",
                                  ms(cg["task_acc"]), ms(cg["concept_acc"]),
                                  ms(cg["task_leak"]), ms(cg["nontask_leak"]),
                                  ms(cg["urr_task"]),
                                  ms(cg["auc_tti"]), ms(cg["nauc_tti"], 4)))
        L.append("")

    L.append("### Notes\n")
    L.append("- MCBM rows: mean±std over seeds 42/43/44 from "
             "`results/summary/aggregate.csv`. CIBM rows: over seeds 42/43/44 from "
             "`../cibm/outputs/summary/cibm_all_runs.csv`.\n"
             "- MCBM also reports leakage(pp)/URR/CKA/DCI/OIS/NIS (see RESULTS.md); "
             "CIBM's public code does not compute those, so they are omitted here.\n"
             "- AUC-TTI/NAUC-TTI use each repo's own intervention protocol "
             "(policy, group definition, scale reconciled) so treat cross-method "
             "TTI as indicative, not identical-definition.")
    with open(OUT, "w") as f:
        f.write("\n".join(L) + "\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
