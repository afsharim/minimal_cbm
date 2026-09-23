"""GPU job queue for the MCBM reproduction runs.

Usage:
  python scripts/run_queue.py jobs.txt [--gpus 0,1,2,3,4,5,6]

jobs.txt: one job per line: "<config_name> <seed>", e.g. "mpi3d-mcbm-3 42".
Runs `bin/train.py <config> -s <seed>` (train + interventions) pinned to one
GPU per job, with:
  - RAM guard: waits until MemAvailable exceeds a per-dataset threshold
  - stagger: minimum delay between consecutive launches of RAM-heavy loaders
  - resume: skips jobs whose results are already complete
  - logs: results/logs/<config>_s<seed>.log, status in results/queue_status.json
"""
import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime

ROOT = "/research/hal-afsharim/minimal_cbm"
PY = "/research/hal-afsharim/miniconda3/envs/mcbm/bin/python"

# Minimum MemAvailable (GB) required to launch, and minimum seconds between
# launches, per dataset prefix. Values reflect measured per-job RSS during
# smoke tests (CIFAR ~33GB, CUB ~20GB, AwA2 ~12GB, MPI3D ~20GB peak from the
# 12.7GB npz loaded per split, Shapes3D ~8GB). MemAvailable must exceed the
# threshold BEFORE launch; the stagger gives a launching job time to allocate
# before the next RAM check.
# Per-job peak RSS (GB) + buffer. The guard requires MemAvailable to exceed
# this before launching, which — with per-dataset staggers giving each job time
# to allocate before the next RAM check — self-limits total concurrency to a
# RAM-safe level (~6-8 jobs on a 125GB host). MPI3D peak halved by the
# transpose-as-view change in disentanglement.py.
RAM_GB = {"mpi3d": 18, "shapes3d": 14, "cifar10": 36, "cub12": 22, "awa2": 15}
STAGGER_S = {"mpi3d": 120, "shapes3d": 60, "cifar10": 90, "cub12": 75, "awa2": 60}

# Per-job VRAM cost (GB), measured from running jobs (~1.7-1.9GB actual for
# the light datasets). Set slightly above actual so two light jobs pack onto a
# card alongside another user's ~6.5GB job (2*2.0+6.5=10.5<11.3). The scheduler
# checks ACTUAL free VRAM (nvidia-smi), so heavy jobs (cub/awa2) wait for the
# shared card to free and pack in then. Host RAM is the real cap for mpi3d
# (~25GB/job) and cifar (~33GB/job).
# Measured actuals: mpi3d/shapes3d/cifar ~1.8-2GB; cub12 (inception 299²) ~8.3GB;
# awa2 (resnet50 224²) ~8GB. Heavy jobs are costed high enough to monopolize an
# 11GB card (no co-location with light jobs, which caused OOM); light jobs pack
# 2-3/GPU (RAM-limited in practice).
VRAM_COST = {"mpi3d": 3.0, "shapes3d": 3.0, "cifar10": 3.0, "cub12": 9.0, "awa2": 8.5}
GPU_CAPACITY = 10.0

# Host RAM is the binding constraint (125GB): per-job peak RSS is large and
# summing many jobs triggers the OOM-killer. Cap total concurrency and per
# dataset so the worst live mix stays within RAM. Peaks (post-optimization,
# approx GB): cifar ~16, mpi3d ~13, shapes3d ~8, cub ~15, awa2 ~10.
MAX_CONCURRENT = 8
DATASET_CAP = {"cifar10": 2, "cub12": 2, "awa2": 2, "mpi3d": 3, "shapes3d": 4}


def mem_available_gb():
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / 1024 / 1024
    return 0.0


def gpu_free_gb(gpu):
    """Actual free VRAM (GB) on a physical GPU, via nvidia-smi. This reflects
    other users' jobs on the shared box, so we never overcommit."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.free",
             "--format=csv,noheader,nounits", "--id=" + str(gpu)],
            timeout=15).decode().strip().splitlines()
        return float(out[0]) / 1024.0
    except Exception:
        return 0.0


GPU_MARGIN_GB = 1.0   # headroom to avoid edge-of-memory OOM
# A launched job may take minutes to actually allocate GPU memory (slow dataset
# __init__: MPI3D reads a 12.7GB npz, Shapes3D filters ~half a million images).
# Until a job's PID shows up in nvidia-smi with GPU memory, we reserve its VRAM
# cost ourselves so the card isn't over-packed. Hard cap so a stuck job can't
# reserve forever.
GPU_ALLOC_CAP_S = 1200


def gpu_allocated_pids():
    """PIDs that currently hold GPU memory (any card)."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid",
             "--format=csv,noheader,nounits"], timeout=15).decode().split()
        return {int(p) for p in out if p.strip().isdigit()}
    except Exception:
        return set()


def n_epochs_of(config):
    import yaml
    subdir = config.split("-")[0]
    with open(os.path.join(ROOT, "configs", subdir, config + ".yaml")) as f:
        cfg = yaml.safe_load(f)
    return cfg["training"]["n_epochs"]


def is_complete(config, seed):
    rdir = os.path.join(ROOT, "results", config, str(seed))
    fm = os.path.join(rdir, "final_metrics.json")
    iv = os.path.join(rdir, "intervention.json")
    if not (os.path.exists(fm) and os.path.exists(iv)):
        return False
    try:
        with open(fm) as f:
            if json.load(f)["epoch"] < n_epochs_of(config):
                return False
        # intervention.json is written incrementally; only trust the
        # 'complete' flag set on the final iteration (or on a clean skip).
        with open(iv) as f:
            return bool(json.load(f).get("complete", False))
    except Exception:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("jobs_file")
    ap.add_argument("--gpus", default="0,1,2,3,4,5,6")
    args = ap.parse_args()

    gpus = [g.strip() for g in args.gpus.split(",") if g.strip() != ""]
    with open(args.jobs_file) as f:
        jobs = [tuple(l.split()) for l in f.read().strip().split("\n")
                if l.strip() and not l.startswith("#")]

    log_dir = os.path.join(ROOT, "results", "logs")
    os.makedirs(log_dir, exist_ok=True)
    status_path = os.path.join(ROOT, "results", "queue_status.json")

    # Single-instance lock: a second queue would double-launch jobs against
    # this one's still-running children and corrupt checkpoints.
    lock_path = os.path.join(ROOT, "results", "queue.lock")
    if os.path.exists(lock_path):
        try:
            old_pid = int(open(lock_path).read().strip())
            os.kill(old_pid, 0)  # raises if not alive
            print(f"Another queue (pid {old_pid}) is running; refusing to start. "
                  f"Remove {lock_path} if stale.")
            sys.exit(2)
        except (ProcessLookupError, ValueError):
            pass  # stale lock, take over
    with open(lock_path, "w") as f:
        f.write(str(os.getpid()))
    import atexit
    atexit.register(lambda: os.path.exists(lock_path) and os.remove(lock_path))

    pending = []
    done, failed = [], []
    for config, seed in jobs:
        if is_complete(config, seed):
            done.append([config, seed, "already-complete"])
            print(f"SKIP {config} s{seed} (complete)")
        else:
            pending.append((config, seed))

    running = []          # list of dicts: gpu, config, seed, proc, logpath, cost
    last_launch = {}      # dataset prefix -> timestamp
    last_gpu_launch = {}  # gpu -> timestamp of most recent launch on it
    retries = {}

    def gpu_load(gpu):
        return sum(r["cost"] for r in running if r["gpu"] == gpu)

    def write_status():
        with open(status_path, "w") as f:
            json.dump({
                "time": datetime.now().isoformat(),
                "pending": [list(j) for j in pending],
                "running": [[r["gpu"], r["config"], r["seed"]] for r in running],
                "done": done,
                "failed": failed,
            }, f, indent=2)

    write_status()
    while pending or running:
        # reap finished
        for r in list(running):
            rc = r["proc"].poll()
            if rc is None:
                continue
            running.remove(r)
            config, seed = r["config"], r["seed"]
            if rc == 0 and is_complete(config, seed):
                done.append([config, seed, "ok"])
                print(f"[{datetime.now():%H:%M}] DONE {config} s{seed} "
                      f"(gpu{r['gpu']})")
            else:
                key = (config, seed)
                retries[key] = retries.get(key, 0) + 1
                if retries[key] <= 1:
                    print(f"[{datetime.now():%H:%M}] RETRY {config} s{seed} rc={rc}")
                    pending.append(key)
                else:
                    failed.append([config, seed, f"rc={rc}"])
                    print(f"[{datetime.now():%H:%M}] FAIL {config} s{seed} rc={rc} "
                          f"see {r['logpath']}")
            write_status()

        # launch: pick the first launchable pending job (no head-of-line
        # blocking), on a GPU with enough ACTUAL free VRAM (nvidia-smi), so we
        # never overcommit against other users' jobs on the shared box.
        now = time.time()
        gpu_free = {g: gpu_free_gb(g) for g in gpus}
        alloc_pids = gpu_allocated_pids()
        # Reserve VRAM for my jobs that haven't allocated GPU memory yet (still
        # in dataset __init__), so nvidia-smi's "free" doesn't fool us into
        # over-packing a card. A job counts as allocated once its PID shows GPU
        # memory; the time cap prevents a stuck job reserving forever.
        def effective_free(g):
            reserved = sum(
                r["cost"] for r in running
                if r["gpu"] == g
                and r["proc"].pid not in alloc_pids
                and now - r["start"] < GPU_ALLOC_CAP_S)
            return gpu_free[g] - reserved

        from collections import Counter
        while pending:
            if len(running) >= MAX_CONCURRENT:
                break
            ds_running = Counter(r["config"].split("-")[0] for r in running)
            pick = None
            for j, (config, seed) in enumerate(pending):
                ds = config.split("-")[0]
                if ds_running[ds] >= DATASET_CAP.get(ds, MAX_CONCURRENT):
                    continue
                cost = VRAM_COST.get(ds, GPU_CAPACITY)
                if mem_available_gb() < RAM_GB.get(ds, 8):
                    continue
                if now - last_launch.get(ds, 0) < STAGGER_S.get(ds, 0):
                    continue
                fit = [g for g in gpus if effective_free(g) - GPU_MARGIN_GB >= cost]
                if not fit:
                    continue
                gpu = max(fit, key=effective_free)
                pick = j
                break
            if pick is None:
                break
            config, seed = pending.pop(pick)
            ds = config.split("-")[0]
            cost = VRAM_COST.get(ds, GPU_CAPACITY)
            # The launched job is appended to `running` below with start≈now, so
            # effective_free() reserves its VRAM on the next pass iteration; no
            # separate local decrement (that would double-count).
            logpath = os.path.join(log_dir, f"{config}_s{seed}.log")
            env = dict(os.environ)
            env.update({
                "CUDA_VISIBLE_DEVICES": gpu,
                "WANDB_MODE": "offline",
                "OMP_NUM_THREADS": "2",
                "MKL_NUM_THREADS": "2",
                # Save full metrics (incl. URR/leakage probe), predictions and a
                # checkpoint at each config save_epochs — intermediate results
                # captured, robust to interruption. (MCBM_SAVE_ONLY_LAST unset.)
            })
            with open(logpath, "a") as lf:
                lf.write(f"\n===== launch {datetime.now()} gpu{gpu} =====\n")
                proc = subprocess.Popen(
                    [PY, os.path.join(ROOT, "bin", "train.py"), config,
                     "-s", str(seed)],
                    cwd=ROOT, env=env, stdout=lf, stderr=subprocess.STDOUT)
            running.append({"gpu": gpu, "config": config, "seed": seed,
                            "proc": proc, "logpath": logpath, "cost": cost,
                            "start": time.time()})
            last_launch[ds] = time.time()
            print(f"[{datetime.now():%H:%M}] LAUNCH {config} s{seed} gpu{gpu} "
                  f"(mem {mem_available_gb():.0f}GB, {len(pending)} pending)")
            write_status()

        time.sleep(20)

    write_status()
    print(f"\nQueue finished: {len(done)} done, {len(failed)} failed")
    for c, s, r in failed:
        print(f"  FAILED {c} s{s}: {r}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
