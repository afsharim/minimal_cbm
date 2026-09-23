"""Collect all MCBM reproduction results into CSVs.

Outputs (results/summary/):
  all_runs.csv            one row per (config, seed) with every scalar metric
  aggregate.csv           mean +/- std across seeds per config
  intervention_curves.csv long format: config, seed, n_groups_intervened, error
"""
import csv
import json
import os
import re
import sys
from collections import defaultdict

import numpy as np
import yaml

ROOT = "/research/hal-afsharim/minimal_cbm"
RESULTS = os.path.join(ROOT, "results")
OUT = os.path.join(RESULTS, "summary")
# Only aggregate finished runs by default (final_metrics is also written at
# intermediate save epochs). Pass --include-partial to include in-progress.
INCLUDE_PARTIAL = "--include-partial" in sys.argv

_NEPOCHS = {}
def n_epochs_of(config):
    if config not in _NEPOCHS:
        subdir = config.split("-")[0]
        try:
            with open(os.path.join(ROOT, "configs", subdir, config + ".yaml")) as f:
                _NEPOCHS[config] = yaml.safe_load(f)["training"]["n_epochs"]
        except Exception:
            _NEPOCHS[config] = None
    return _NEPOCHS[config]

MODEL_LABEL = {
    "vanilla": "Vanilla", "cbm": "CBM", "cem": "CEM",
    "arhcbm": "ARCBM", "shcbm": "SCBM",
}


def parse_config_name(config):
    ds, rest = config.split("-", 1)
    if rest.startswith("mcbm"):
        g = rest.split("-", 1)[1]
        gamma = {"005": 0.05, "01": 0.1, "03": 0.3, "05": 0.5,
                 "1": 1.0, "3": 3.0, "5": 5.0}.get(g, g)
        level = None
        order = {"mpi3d": [1.0, 3.0, 5.0], "shapes3d": [1.0, 3.0, 5.0],
                 "cifar10": [0.1, 0.3, 0.5], "cub12": [0.05, 0.1, 0.3],
                 "awa2": [0.05, 0.1, 0.3]}
        if ds in order and gamma in order[ds]:
            level = ["low", "medium", "high"][order[ds].index(gamma)]
        return ds, f"MCBM ({level} g)" if level else f"MCBM (g={gamma})", gamma
    return ds, MODEL_LABEL.get(rest, rest), None


def main():
    os.makedirs(OUT, exist_ok=True)
    runs = []
    curves = []
    for config in sorted(os.listdir(RESULTS)):
        cdir = os.path.join(RESULTS, config)
        if not os.path.isdir(cdir) or config in ("logs", "summary"):
            continue
        if "smoke" in config:
            continue
        for seed in sorted(os.listdir(cdir)):
            rdir = os.path.join(cdir, seed)
            fm = os.path.join(rdir, "final_metrics.json")
            if not os.path.exists(fm):
                continue
            with open(fm) as f:
                m = json.load(f)
            # Skip in-progress runs (intermediate save epoch) unless asked.
            ne = n_epochs_of(config)
            if not INCLUDE_PARTIAL and ne is not None and m.get("epoch", 0) < ne:
                continue
            ds, model, gamma = parse_config_name(config)
            row = {
                "config": config, "dataset": ds, "model": model,
                "gamma": gamma, "seed": int(seed),
                "epoch": m.get("epoch"),
                "task_acc": m.get("accuracy_test/task"),
                "concept_acc": m.get("accuracy_test/concepts"),
                "urr_task": None if m.get("mutual_inf_test/nuisances_task") is None
                    else 100 * m["mutual_inf_test/nuisances_task"],
                "urr_nontask": None if m.get("mutual_inf_test/nuisances_nontask") is None
                    else 100 * m["mutual_inf_test/nuisances_nontask"],
                "leak_acc_task": m.get("accuracy_test/nuisances_task"),
                "leak_acc_nontask": m.get("accuracy_test/nuisances_nontask"),
                "ece": m.get("calibration_test/ece"),
                "brier": m.get("calibration_test/brier"),
                "map_concepts": m.get("map_test/concepts"),
            }
            iv = os.path.join(rdir, "intervention.json")
            if os.path.exists(iv):
                with open(iv) as f:
                    ivd = json.load(f)
                if not ivd.get("skipped") and ivd.get("errors"):
                    def tti(errs):
                        accs = [100 - e for e in errs]
                        n = len(accs) - 1
                        if n <= 0:
                            return None, None
                        return (float(np.mean(accs[1:])),
                                float((accs[-1] - accs[0]) / 100.0 / n))
                    errs = ivd["errors"]                       # lowest-confidence
                    errs_r = ivd.get("errors_random", errs)    # random policy
                    for i, e in enumerate(errs):
                        curves.append({"config": config, "dataset": ds,
                                       "model": model, "seed": int(seed),
                                       "n_groups_intervened": i, "error": e,
                                       "error_random": errs_r[i] if i < len(errs_r) else None})
                    row["auc_tti"], row["nauc_tti"] = tti(errs)
                    row["auc_tti_random"], row["nauc_tti_random"] = tti(errs_r)
                    row["interv_error_0"] = errs[0]
                    row["interv_error_full"] = errs[-1]
            runs.append(row)

    cols = ["config", "dataset", "model", "gamma", "seed", "epoch",
            "task_acc", "concept_acc", "urr_task", "urr_nontask",
            "leak_acc_task", "leak_acc_nontask", "ece", "brier",
            "map_concepts", "auc_tti", "nauc_tti",
            "auc_tti_random", "nauc_tti_random",
            "interv_error_0", "interv_error_full"]
    with open(os.path.join(OUT, "all_runs.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in runs:
            w.writerow({c: r.get(c) for c in cols})

    # aggregate mean +/- std across seeds
    groups = defaultdict(list)
    for r in runs:
        groups[r["config"]].append(r)
    agg_rows = []
    metric_cols = cols[6:]
    for config, rs in sorted(groups.items()):
        ds, model, gamma = parse_config_name(config)
        row = {"config": config, "dataset": ds, "model": model,
               "gamma": gamma, "n_seeds": len(rs),
               "seeds": "/".join(str(r["seed"]) for r in rs)}
        for mc in metric_cols:
            vals = [r[mc] for r in rs if r.get(mc) is not None]
            if vals:
                row[f"{mc}_mean"] = round(float(np.mean(vals)), 4)
                row[f"{mc}_std"] = round(float(np.std(vals)), 4)
        agg_rows.append(row)
    agg_cols = ["config", "dataset", "model", "gamma", "n_seeds", "seeds"] + \
        [f"{m}_{s}" for m in metric_cols for s in ("mean", "std")]
    with open(os.path.join(OUT, "aggregate.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=agg_cols)
        w.writeheader()
        for r in agg_rows:
            w.writerow({c: r.get(c) for c in agg_cols})

    with open(os.path.join(OUT, "intervention_curves.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["config", "dataset", "model", "seed",
                                          "n_groups_intervened", "error",
                                          "error_random"])
        w.writeheader()
        for r in curves:
            w.writerow(r)

    print(f"{len(runs)} runs, {len(agg_rows)} configs -> {OUT}")


if __name__ == "__main__":
    main()
