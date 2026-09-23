"""Auto-refresh the results file whenever new runs complete.

Polls the completed-run count every POLL_SECONDS. When it increases, it
regenerates (in order):
  1. metrics_history snapshots for all runs   (backfill_snapshots.py)
  2. results/summary/*.csv                      (collect_results.py)
  3. results/summary/RESULTS.md                 (make_results_md.py)

Appends a heartbeat line to results/logs/results_watch.log each cycle. When all
120 runs are complete it does one final refresh and exits. Safe to run alongside
the training queue (read-only w.r.t. run dirs; only writes results/summary + logs
+ per-run metrics_history).
"""
import json
import os
import subprocess
import time

import yaml

ROOT = "/research/hal-afsharim/minimal_cbm"
PY = "/research/hal-afsharim/miniconda3/envs/mcbm/bin/python"
RESULTS = os.path.join(ROOT, "results")
LOG = os.path.join(RESULTS, "logs", "results_watch.log")
POLL_SECONDS = int(os.environ.get("MCBM_WATCH_POLL", "300"))
TOTAL_RUNS = 120

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


def count_complete():
    """Runs whose final_metrics.json reached the configured n_epochs."""
    done = 0
    by_ds = {}
    for config in os.listdir(RESULTS):
        cdir = os.path.join(RESULTS, config)
        if not os.path.isdir(cdir) or config in ("logs", "summary"):
            continue
        ds = config.split("-")[0]
        for seed in os.listdir(cdir):
            fm = os.path.join(cdir, seed, "final_metrics.json")
            if not os.path.exists(fm):
                continue
            try:
                e = json.load(open(fm)).get("epoch", 0)
            except Exception:
                continue
            ne = n_epochs_of(config)
            if ne and e >= ne:
                done += 1
                by_ds[ds] = by_ds.get(ds, 0) + 1
    return done, by_ds


def log(msg):
    os.makedirs(os.path.dirname(LOG), exist_ok=True)
    with open(LOG, "a") as f:
        f.write(msg + "\n")
    print(msg, flush=True)


def regenerate():
    # posthoc_metrics is incremental (only new complete runs) and CPU-heavy, so
    # cap its BLAS threads/workers to avoid starving the training queue.
    env = dict(os.environ)
    env.update({"OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2",
                "OPENBLAS_NUM_THREADS": "2", "MCBM_POSTHOC_WORKERS": "6"})
    for script in ("backfill_snapshots.py", "collect_results.py",
                   "posthoc_metrics.py", "make_results_md.py"):
        try:
            r = subprocess.run([PY, os.path.join(ROOT, "scripts", script)],
                               cwd=ROOT, capture_output=True, text=True,
                               timeout=7200, env=env)
            tail = (r.stdout or r.stderr).strip().splitlines()
            note = tail[-1] if tail else "(no output)"
            log(f"    {script}: {note}")
        except Exception as exc:  # keep the watcher alive no matter what
            log(f"    {script}: ERROR {exc!r}")


def main():
    last = -1
    log(f"[watch] started, polling every {POLL_SECONDS}s "
        f"(pid {os.getpid()})")
    while True:
        done, by_ds = count_complete()
        if done != last:
            log(f"[{time.strftime('%m-%d %H:%M')}] complete={done}/{TOTAL_RUNS} "
                f"{by_ds} -> refreshing result file")
            regenerate()
            last = done
        if done >= TOTAL_RUNS:
            log(f"[watch] all {TOTAL_RUNS} runs complete; final refresh done, "
                f"exiting.")
            break
        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    main()
