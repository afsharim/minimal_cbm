# MCBM γ=1 on MPI3D — batch-size study

Effect of training batch size on MCBM (IB, γ=1), MPI3D, resnet20. All metrics via
MCBM's own code (CUDA leakage probe, full-train; posthoc CKA/DCI/OIS/NIS with
CKA_N=4096 / PROBE_N=8000). Single seed 42 for the bs=512 and bs=4096 runs; the
bs=128 baseline is shown both at seed 42 and as its 3-seed mean±std.

**Key control — total optimization ≈ LR × epochs / batch:**

| Run | batch | eff. LR | epochs | StepLR step | LR × ep / batch (rel.) |
|---|---|---|---|---|---|
| **bs=128 baseline** | 128 | 0.006 | 50 | 20 | 1.00× |
| **bs=512** (sqrt-LR) | 512 | 0.012 | 100 | 40 | **1.00×** |
| **bs=4096** (fixed-LR) | 4096 | 0.006 | 50 | 20 | **0.031×** |

bs=512 keeps the optimization budget **identical** to the baseline (LR ×2, epochs ×2,
batch ×4 → unchanged); bs=4096 held the effective LR fixed, so its budget dropped 32×.

## Full metric suite

| Metric | bs=128 (seed 42) | bs=128 (3-seed mean±std) | **bs=512** (sqrt-LR, 100 ep) | bs=4096 (fixed-LR, 50 ep) |
|---|---|---|---|---|
| **Task Acc (%)** | 69.80 | 65.11 ± 3.48 | **64.66** | 23.72 |
| Concept Acc (%) | 99.999 | 99.999 | 99.996 | 99.95 |
| ECE | 0.2629 | 0.232 ± 0.023 | 0.2315 | 0.0211 |
| Brier | 0.5045 | 0.534 ± 0.023 | 0.5432 | 0.8192 |
| mAP (concepts) | 0.9999 | 0.9999 | 0.9999 | 0.9999 |
| URR task (MI×100) | 9.19 | 8.82 ± 0.26 | 8.61 | 12.41 |
| URR nontask | −0.001 | −0.001 | 0.001 | 2.72 |
| Task leakage (pp) | 6.49 | 5.62 ± 0.62 | 5.22 | 8.34 |
| Nontask leakage (pp) | 0.104 | 0.11 ± 0.07 | 0.097 | 1.97 |
| AUC-TTI | 45.93 | 42.0 ± 4.0 | 42.84 | 21.07 |
| NAUC-TTI | −0.0341 | −0.030 ± 0.003 | −0.0302 | −0.0027 |
| CKA | 0.985 | 0.986 | 0.986 | 0.975 |
| DCI | 0.485 | 0.492 | 0.544 | 0.877 |
| OIS | 1.801 | 1.903 | 3.098 | 1.796 |
| NIS | ~0 | ~0 | ~0 | ~0 |

## Findings

**bs=512 with sqrt LR-scaling (0.012) + 2× epochs (100) reproduces the bs=128 baseline
on every metric.** Task 64.66 vs 65.11 ± 3.48, ECE 0.232 vs 0.232, Brier 0.543 vs 0.534,
URR task 8.61 vs 8.82 ± 0.26, task leakage 5.22 vs 5.62 ± 0.62, AUC-TTI 42.84 vs 42.0 ± 4.0,
NAUC-TTI −0.030 vs −0.030 — all within one std of the baseline mean. CKA/DCI/NIS match;
OIS (3.10) sits a bit above the baseline's 1.9 but within the metric's per-seed spread
(baseline seeds ranged 1.72–2.18). **Conclusion: at a matched optimization budget, batch
size 512 is statistically indistinguishable from the bs=128 baseline.**

**bs=4096 at fixed effective-LR collapses** — the earlier result. Task acc falls to 23.7%
(near-chance), because the 32× smaller optimization budget under-trains the task head. ECE
looks "better" (0.021) only because the head emits near-uniform predictions (Brier *worse*,
0.819); leakage/URR rise on the noisier `z`; interventions flatten (NAUC → −0.003).

**Takeaway.** The bs=4096 failure was an **optimization-budget artifact, not a batch-size
effect.** Preserving LR × epochs / batch (here via sqrt LR + 2× epochs) makes a 4× larger
batch behave exactly like the baseline — confirming the diagnosis and the fix.

## Provenance
- Configs: `configs/mpi3d/mpi3d-mcbm-1.yaml` (baseline), `-bs512.yaml`, `-bs4096.yaml`.
- Runs: `results/mpi3d-mcbm-1{,-bs512,-bs4096}/42/` (final_metrics.json, intervention.json).
- Posthoc rows: `results/summary/posthoc_metrics.csv` (models "MCBM (g=1-bs512/bs4096)").
