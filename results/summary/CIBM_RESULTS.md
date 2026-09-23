# CIBM Results (dsb-ifi/cibm) on our datasets


**45 runs collected.**

Concepts' Information Bottleneck Models (ICLR 2026, arXiv:2602.14626) run on the same datasets as our MCBM reproduction. Frozen backbones (InceptionV3 for CUB, xlsa17 ResNet-101 for AwA2, ImageNet ResNet-50 for CIFAR-10/MPI3D/Shapes3D). Seeds 42/43/44. Three IB variants from the repo's own recipe. Accuracies in %, AUC-TTI = mean of the test-time-intervention curve (%), NAUC-TTI = normalized intervention AUC (CIBM convention).

## CUB

| Variant | Seeds | Task Acc | Concept Acc | AUC-TTI | NAUC-TTI |
|---|---|---|---|---|---|
| Basic CBM | 3 | 42.94 ± 0.28 | 89.65 ± 0.01 | 65.13 ± 0.44 | 0.0112 ± 0.0003 |
| IB-CBM (variational) | 3 | 55.37 ± 0.45 | 89.65 ± 0.00 | 74.99 ± 1.25 | 0.0141 ± 0.0005 |
| IB-CBM (entropy H(C)) | 3 | 55.15 ± 0.44 | 89.67 ± 0.00 | 74.87 ± 0.96 | 0.0141 ± 0.0006 |

## AwA2

| Variant | Seeds | Task Acc | Concept Acc | AUC-TTI | NAUC-TTI |
|---|---|---|---|---|---|
| Basic CBM | 3 | 87.11 ± 0.05 | 96.87 ± 0.00 | 84.54 ± 0.77 | 0.0029 ± 0.0015 |
| IB-CBM (variational) | 3 | 88.53 ± 0.15 | 96.87 ± 0.00 | 92.96 ± 0.15 | 0.0059 ± 0.0003 |
| IB-CBM (entropy H(C)) | 3 | 88.47 ± 0.12 | 96.88 ± 0.00 | 92.93 ± 0.41 | 0.0060 ± 0.0003 |

## CIFAR-10

| Variant | Seeds | Task Acc | Concept Acc | AUC-TTI | NAUC-TTI |
|---|---|---|---|---|---|
| Basic CBM | 3 | 91.12 ± 0.07 | 87.17 ± 0.01 | 75.84 ± 0.28 | -0.0045 ± 0.0001 |
| IB-CBM (variational) | 3 | 91.14 ± 0.11 | 87.18 ± 0.01 | 62.07 ± 0.52 | -0.0038 ± 0.0001 |
| IB-CBM (entropy H(C)) | 3 | 91.01 ± 0.15 | 87.19 ± 0.01 | 62.23 ± 0.65 | -0.0037 ± 0.0001 |

## MPI3D

| Variant | Seeds | Task Acc | Concept Acc | AUC-TTI | NAUC-TTI |
|---|---|---|---|---|---|
| Basic CBM | 3 | 73.53 ± 1.01 | 98.78 ± 0.06 | 37.65 ± 0.54 | -0.0406 ± 0.0007 |
| IB-CBM (variational) | 3 | 87.84 ± 0.11 | 98.77 ± 0.04 | 24.01 ± 0.53 | -0.0506 ± 0.0004 |
| IB-CBM (entropy H(C)) | 3 | 87.74 ± 0.14 | 98.85 ± 0.03 | 23.97 ± 0.40 | -0.0504 ± 0.0004 |

## Shapes3D

| Variant | Seeds | Task Acc | Concept Acc | AUC-TTI | NAUC-TTI |
|---|---|---|---|---|---|
| Basic CBM | 3 | 78.89 ± 0.16 | 93.88 ± 0.01 | 44.85 ± 0.12 | -0.0297 ± 0.0001 |
| IB-CBM (variational) | 3 | 99.94 ± 0.03 | 94.79 ± 0.01 | 38.40 ± 1.61 | -0.0416 ± 0.0002 |
| IB-CBM (entropy H(C)) | 3 | 99.90 ± 0.04 | 95.03 ± 0.01 | 41.43 ± 7.95 | -0.0416 ± 0.0002 |

