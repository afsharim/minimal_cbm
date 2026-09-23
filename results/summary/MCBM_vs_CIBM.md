# MCBM vs CIBM — side-by-side on the same datasets

Our reproduction of *There Was Never a Bottleneck in CBMs* (arXiv:2506.04877; **MCBM**) versus *Concepts' Information Bottleneck Models* (arXiv:2602.14626, dsb-ifi/cibm; **CIBM**), run on the same datasets, seeds 42/43/44. Metrics both pipelines produce: Task Acc, Concept Acc (%), AUC-TTI (mean intervention-curve accuracy, %), NAUC-TTI (normalized intervention AUC).

> **Read this first — the setups are not backbone-matched.** MCBM trains the image encoder **end-to-end**; CIBM trains an MLP head on **frozen** backbone embeddings (InceptionV3 for CUB, xlsa17 ResNet-101 for AwA2, ImageNet ResNet-50 for CIFAR/MPI3D/Shapes3D). On the synthetic datasets a trained encoder reaches ~100% task acc while frozen ImageNet features cannot, so **task-accuracy gaps on MPI3D/Shapes3D are dominated by the backbone difference, not the method.** The informative within-method signal is how each paper's IB regularizer moves the numbers relative to its own baseline CBM.

> **Opposing IB behaviour (the key finding).** CIBM's IB regularizer *raises* task accuracy over its Basic CBM (e.g. Shapes3D 78.9→99.9, MPI3D 73.5→87.8); MCBM's γ-bottleneck *lowers* it as γ grows (MPI3D CBM 100 → mcbm-γ 65→25). The two 'information bottleneck' recipes push task accuracy in opposite directions.

## MPI3D

| Method | Task Acc | Concept Acc | AUC-TTI | NAUC-TTI |
|---|---|---|---|---|
| MCBM · CBM (trained enc.) | 99.96 ± 0.01 | 100.00 ± 0.00 | 49.61 ± 2.39 | -0.0592 ± 0.0005 |
| CIBM · Basic CBM (frozen emb.) | 73.53 ± 1.01 | 98.78 ± 0.06 | 37.65 ± 0.54 | -0.0406 ± 0.0007 |
| MCBM · γ low g | 65.11 ± 3.48 | 100.00 ± 0.00 | 42.00 ± 3.99 | -0.0302 ± 0.0029 |
| MCBM · γ medium g | 24.71 ± 0.54 | 100.00 ± 0.00 | 24.63 ± 0.51 | -0.0001 ± 0.0000 |
| MCBM · γ high g | 24.73 ± 0.32 | 100.00 ± 0.00 | 24.61 ± 0.37 | -0.0001 ± 0.0001 |
| CIBM · IB-CBM (variational) (frozen emb.) | 87.84 ± 0.11 | 98.77 ± 0.04 | 24.01 ± 0.53 | -0.0506 ± 0.0004 |
| CIBM · IB-CBM (entropy) (frozen emb.) | 87.74 ± 0.14 | 98.85 ± 0.03 | 23.97 ± 0.40 | -0.0504 ± 0.0004 |

## Shapes3D

| Method | Task Acc | Concept Acc | AUC-TTI | NAUC-TTI |
|---|---|---|---|---|
| MCBM · CBM (trained enc.) | 100.00 ± 0.00 | 100.00 ± 0.00 | 72.82 ± 3.80 | -0.0416 ± 0.0002 |
| CIBM · Basic CBM (frozen emb.) | 78.89 ± 0.16 | 93.88 ± 0.01 | 44.85 ± 0.12 | -0.0297 ± 0.0001 |
| MCBM · γ low g | 74.60 ± 4.78 | 96.67 ± 2.36 | 49.99 ± 5.47 | -0.0254 ± 0.0026 |
| MCBM · γ medium g | 30.40 ± 0.33 | 96.67 ± 2.36 | 29.93 ± 0.25 | -0.0004 ± 0.0001 |
| MCBM · γ high g | 29.92 ± 0.29 | 96.67 ± 2.36 | 29.88 ± 0.29 | -0.0001 ± 0.0000 |
| CIBM · IB-CBM (variational) (frozen emb.) | 99.94 ± 0.03 | 94.79 ± 0.01 | 38.40 ± 1.61 | -0.0416 ± 0.0002 |
| CIBM · IB-CBM (entropy) (frozen emb.) | 99.90 ± 0.04 | 95.03 ± 0.01 | 41.43 ± 7.95 | -0.0416 ± 0.0002 |

## CIFAR-10

| Method | Task Acc | Concept Acc | AUC-TTI | NAUC-TTI |
|---|---|---|---|---|
| MCBM · CBM (trained enc.) | 73.78 ± 0.16 | 84.74 ± 0.17 | 74.00 ± 1.33 | -0.0014 ± 0.0003 |
| CIBM · Basic CBM (frozen emb.) | 91.12 ± 0.07 | 87.17 ± 0.01 | 75.84 ± 0.28 | -0.0045 ± 0.0001 |
| MCBM · γ low g | 72.22 ± 0.19 | 84.91 ± 0.20 | 75.28 ± 0.25 | 0.0004 ± 0.0001 |
| MCBM · γ medium g | 70.05 ± 0.59 | 84.87 ± 0.21 | 74.52 ± 0.35 | 0.0008 ± 0.0001 |
| MCBM · γ high g | 69.39 ± 0.56 | 84.89 ± 0.20 | 74.36 ± 0.22 | 0.0009 ± 0.0001 |
| CIBM · IB-CBM (variational) (frozen emb.) | 91.14 ± 0.11 | 87.18 ± 0.01 | 62.07 ± 0.52 | -0.0038 ± 0.0001 |
| CIBM · IB-CBM (entropy) (frozen emb.) | 91.01 ± 0.15 | 87.19 ± 0.01 | 62.23 ± 0.65 | -0.0037 ± 0.0001 |

## CUB

| Method | Task Acc | Concept Acc | AUC-TTI | NAUC-TTI |
|---|---|---|---|---|
| MCBM · CBM (trained enc.) | 77.30 ± 0.59 | 96.36 ± 0.12 | 78.14 ± 1.28 | -0.0040 ± 0.0016 |
| CIBM · Basic CBM (frozen emb.) | 42.94 ± 0.28 | 89.65 ± 0.01 | 65.13 ± 0.44 | 0.0112 ± 0.0003 |
| MCBM · γ low g | 77.52 ± 0.76 | 95.96 ± 0.93 | 84.56 ± 1.24 | 0.0065 ± 0.0012 |
| MCBM · γ medium g | 76.92 ± 0.42 | 96.00 ± 0.91 | 85.57 ± 0.70 | 0.0080 ± 0.0010 |
| MCBM · γ high g | 75.16 ± 0.73 | 95.93 ± 0.89 | 84.56 ± 0.48 | 0.0095 ± 0.0013 |
| CIBM · IB-CBM (variational) (frozen emb.) | 55.37 ± 0.45 | 89.65 ± 0.00 | 74.99 ± 1.25 | 0.0141 ± 0.0005 |
| CIBM · IB-CBM (entropy) (frozen emb.) | 55.15 ± 0.44 | 89.67 ± 0.00 | 74.87 ± 0.96 | 0.0141 ± 0.0006 |

## AwA2

| Method | Task Acc | Concept Acc | AUC-TTI | NAUC-TTI |
|---|---|---|---|---|
| MCBM · CBM (trained enc.) | 94.40 ± 0.05 | 98.90 ± 0.01 | 94.80 ± 0.83 | -0.0040 ± 0.0018 |
| CIBM · Basic CBM (frozen emb.) | 87.11 ± 0.05 | 96.87 ± 0.00 | 84.54 ± 0.77 | 0.0029 ± 0.0015 |
| MCBM · γ low g | 94.19 ± 0.01 | 98.95 ± 0.02 | 94.54 ± 1.13 | -0.0019 ± 0.0005 |
| MCBM · γ medium g | 94.12 ± 0.19 | 98.96 ± 0.03 | 93.68 ± 0.99 | -0.0019 ± 0.0006 |
| MCBM · γ high g | 93.96 ± 0.11 | 98.97 ± 0.02 | 94.14 ± 0.33 | -0.0014 ± 0.0001 |
| CIBM · IB-CBM (variational) (frozen emb.) | 88.53 ± 0.15 | 96.87 ± 0.00 | 92.96 ± 0.15 | 0.0059 ± 0.0003 |
| CIBM · IB-CBM (entropy) (frozen emb.) | 88.47 ± 0.12 | 96.88 ± 0.00 | 92.93 ± 0.41 | 0.0060 ± 0.0003 |

### Notes

- MCBM rows: mean±std over seeds 42/43/44 from `results/summary/aggregate.csv`. CIBM rows: over seeds 42/43/44 from `../cibm/outputs/summary/cibm_all_runs.csv`.
- MCBM also reports leakage(pp)/URR/CKA/DCI/OIS/NIS (see RESULTS.md); CIBM's public code does not compute those, so they are omitted here.
- AUC-TTI/NAUC-TTI use each repo's own intervention protocol (policy, group definition, scale reconciled) so treat cross-method TTI as indicative, not identical-definition.
