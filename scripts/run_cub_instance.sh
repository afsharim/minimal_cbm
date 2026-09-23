#!/bin/bash
# Re-run all CUB experiments EXACTLY as MCBM: instance-level attributes
# (CUB_processed/base pkls + modify_pkls=True), 8 configs x 3 seeds = 24 runs,
# ~5 concurrent across the 7 GPUs (CUB ~22GB RAM each; probe is small here).
set -u
PY=/research/hal-afsharim/miniconda3/envs/mcbm/bin/python
cd /research/hal-afsharim/minimal_cbm
mkdir -p results/logs/cub_rerun
K=7; NGPU=7
configs=(cub12-vanilla cub12-cbm cub12-cem cub12-arhcbm cub12-shcbm \
         cub12-mcbm-005 cub12-mcbm-01 cub12-mcbm-03)
jobs=()
for c in "${configs[@]}"; do for s in 42 43 44; do jobs+=("$c|$s"); done; done

run_one() {
  IFS="|" read -r c s <<< "$1"
  local gpu=$(( $2 % NGPU ))
  echo "[$(date +%H:%M:%S)] START $c s$s gpu$gpu"
  CUDA_VISIBLE_DEVICES=$gpu WANDB_MODE=offline \
    OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6 NUMEXPR_NUM_THREADS=6 \
    "$PY" bin/train.py "$c" -s "$s" \
    > "results/logs/cub_rerun/${c}_s${s}.log" 2>&1
  echo "[$(date +%H:%M:%S)] DONE  $c s$s rc=$?"
}

i=0
for j in "${jobs[@]}"; do
  run_one "$j" "$i" &
  i=$((i+1))
  while [ "$(jobs -rp | wc -l)" -ge "$K" ]; do sleep 10; done
done
wait
echo "[$(date +%H:%M:%S)] ALL ${#jobs[@]} CUB RUNS DONE"
