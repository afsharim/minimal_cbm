"""Materialize preserved per-checkpoint metric snapshots for existing runs.

For every results/<config>/<seed>/train_log.jsonl this writes
results/<config>/<seed>/metrics_history/epoch_XXXX.json for each epoch that is
a multiple of SNAP_EVERY (default 10) plus the last logged epoch. This mirrors
what train.py now writes live, so runs launched before that change (and runs
still in flight) also keep the full metric history as one file per checkpoint.

Idempotent and safe to re-run at any time (e.g. after more epochs are logged).
Cheap metrics are present for every epoch; the expensive URR/leakage keys are
only present on save-epochs (that is exactly what train.py writes live too).
"""
import json
import os
import sys

ROOT = "/research/hal-afsharim/minimal_cbm"
RESULTS = os.path.join(ROOT, "results")
SNAP_EVERY = int(os.environ.get("MCBM_SNAPSHOT_EVERY", "10"))


def backfill_one(rdir):
    log_path = os.path.join(rdir, "train_log.jsonl")
    if not os.path.exists(log_path):
        return 0
    rows = []
    with open(log_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    if not rows:
        return 0
    # Keep the last line seen for each epoch (a save-epoch line carries the rich
    # URR keys and should win over any earlier plain line for that epoch).
    by_epoch = {}
    for r in rows:
        e = r.get("epoch")
        if e is None:
            continue
        prev = by_epoch.get(e)
        # Prefer the row with more keys (the save-epoch/URR row).
        if prev is None or len(r) >= len(prev):
            by_epoch[e] = r
    if not by_epoch:
        return 0
    last_epoch = max(by_epoch)
    snap_dir = os.path.join(rdir, "metrics_history")
    os.makedirs(snap_dir, exist_ok=True)
    written = 0
    for e, r in sorted(by_epoch.items()):
        if not (SNAP_EVERY > 0 and (e % SNAP_EVERY == 0 or e == last_epoch)):
            continue
        out = os.path.join(snap_dir, f"epoch_{e:04d}.json")
        with open(out, "w") as f:
            json.dump(r, f, indent=2)
        written += 1
    return written


def main():
    total_runs = total_files = 0
    for config in sorted(os.listdir(RESULTS)):
        cdir = os.path.join(RESULTS, config)
        if not os.path.isdir(cdir) or config in ("logs", "summary"):
            continue
        for seed in sorted(os.listdir(cdir)):
            rdir = os.path.join(cdir, seed)
            if not os.path.isdir(rdir):
                continue
            n = backfill_one(rdir)
            if n:
                total_runs += 1
                total_files += n
                print(f"  {config}/{seed}: {n} snapshots -> metrics_history/")
    print(f"backfilled {total_files} snapshot files across {total_runs} runs "
          f"(cadence every {SNAP_EVERY} epochs)")


if __name__ == "__main__":
    main()
