# MCBM γ=1 on MPI3D — large-batch (bs=4096) run

**Config:** `configs/mpi3d/mpi3d-mcbm-1-bs4096.yaml` — MCBM (IB, γ=1), encoder resnet20,
50 epochs, SGD momentum 0.9, StepLR (step 20, γ 0.1). **Single seed 42, GPU 0.**

**Learning rate:** effective LR pinned to **0.006** (`base_lr 0.006 × 4096/4096`), i.e. the
*same* effective LR as the bs=128 baseline — so the 32× batch increase does **not** change the
step size. Training was stable (no divergence).

Baseline column = `mpi3d-mcbm-1` (bs=128), **seed 42** for a like-for-like comparison; the
bs=128 3-seed mean±std is given underneath for context. All metrics computed with MCBM's own
code (CUDA leakage probe; posthoc CKA/DCI/OIS/NIS with CKA_N=4096 / PROBE_N=8000).

## Full metric suite (all metrics MCBM reports)

| Metric | bs=128 (seed 42) | **bs=4096 (seed 42)** | Δ |
|---|---|---|---|
| Task Acc (%) | 69.80 | **23.72** | **−46.1** |
| Concept Acc (%) | 99.999 | 99.95 | −0.05 |
| ECE | 0.2629 | 0.0211 | −0.242 |
| Brier | 0.5045 | 0.8192 | +0.315 |
| mAP (concepts) | 0.9999 | 0.9999 | ~0 |
| URR task (MI×100) | 9.19 | 12.41 | +3.22 |
| URR nontask (MI×100) | −0.001 | 2.72 | +2.72 |
| Task leakage (pp) | 6.49 | 8.34 | +1.85 |
| Nontask leakage (pp) | 0.104 | 1.97 | +1.87 |
| AUC-TTI | 45.93 | 21.07 | −24.9 |
| NAUC-TTI | −0.0341 | −0.0027 | +0.031 |
| CKA | 0.985 | 0.975 | −0.010 |
| DCI (disentanglement) | 0.485 | 0.877 | +0.392 |
| OIS | 1.801 | 1.796 | −0.005 |
| NIS | ~0 | ~0 | ~0 |

**bs=128 baseline, 3-seed mean±std (context):** Task 65.11±3.48 · Concept 99.999 ·
URR task 8.82±0.26 · URR nontask −0.001 · Task leak 5.62±0.62 · Nontask leak 0.11±0.07 ·
ECE 0.232±0.023 · Brier 0.534±0.023 · mAP 0.9999 · AUC-TTI 42.0±4.0 · NAUC-TTI −0.030±0.003.

## What happened

At bs=4096 with the **effective LR held at 0.006**, the model does ~**32× fewer gradient
updates** over the same 50 epochs (bs=128 → thousands of iters/epoch; bs=4096 → tens). Same
step size × far fewer steps = far less total optimization. Result:

- **Task head under-trains → collapses to near-chance.** Task acc 69.8% → **23.7%**; test
  task cross-entropy 1.75 ≈ ln(#classes), i.e. barely better than a uniform predictor.
- **Concepts are unaffected (99.95%).** They are ~linearly separable from `z`, so they converge
  in very few updates regardless of batch size.
- **ECE "improves" but is misleading.** ECE drops to 0.021 only because the task head outputs
  near-uniform low-confidence predictions — Brier gets *worse* (0.505 → 0.819). It is trivially
  "calibrated" by being unconfident, not by being right.
- **Leakage / URR rise** (task leak 6.5→8.3 pp, URR 9.2→12.4; nontask leak 0.10→1.97 pp): an
  under-trained, noisier `z` carries more spurious nuisance information.
- **Interventions flatten** (NAUC-TTI −0.034 → −0.003): with the task head barely using the
  concepts, replacing predicted concepts with ground truth changes little.
- **DCI rises (0.49 → 0.88):** because the broken task head smuggles little task-signal into the
  concept dimensions, `z` looks more cleanly "one dimension per concept."

## If the goal is bs=4096 that *matches* the baseline

Two standard large-batch remedies (not run here — say the word):

1. **Linear LR scaling (Goyal et al. 2017).** The repo's *default* scaling would set
   `lr = base_lr × bs/base_bs = 0.003 × 4096/64 = 0.192` (+ a short warmup). This is the
   canonical recipe for keeping large-batch training on par with small-batch. (It was overridden
   to 0.006 here per the "fix effective LR" choice.)
2. **More epochs / more updates.** Keep LR 0.006 but train ~8–16× longer (e.g. 50 → 400–800
   epochs), optionally with warmup, to match the bs=128 update count.
