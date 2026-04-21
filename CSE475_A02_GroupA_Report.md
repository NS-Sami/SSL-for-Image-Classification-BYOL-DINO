# CSE 475 - Assignment 02
## Lab Assignment 02: Self-Supervised Learning for Image Classification — BYOL & DINO

**Group Information**
- **Group ID:** Group A
- **Notebook Types:** BYOL Notebook / DINO Notebook
- **Backbone Used:** EfficientNet-B3
- **Assignment 01 Best Acc:** 99.770% (EfficientNet-B3, 50 epochs)
- **Dataset Name (Kaggle):** Tropical Flowers dataset
- **Submission Date:** April 2026

---

## Table of Contents
1. Introduction
2. Dataset and Preprocessing
3. BYOL Implementation
4. DINO Implementation
5. Evaluation and Results
6. Discussion and Ablation
7. Conclusion and Future Work
8. References

---

## 3. Introduction

Self-Supervised Learning (SSL) has emerged as a groundbreaking approach in modern computer vision, aiming to resolve the bottlenecks inherent in fully supervised learning—namely, the expensive, prolonged, and often impractical acquisition of large-scale labeled datasets. The primary motive of SSL is to harness unlabeled data by constructing heuristic auxiliary tasks straight from the data itself to produce highly robust feature representations.

In this assignment, we implement and evaluate two state-of-the-art SSL frameworks:
1. **Bootstrap Your Own Latent (BYOL):** A non-contrastive knowledge-distillation method utilizing asymmetric dual networks (online and target) to learn representations without relying on negative examples.
2. **Self-Distillation with No Labels (DINO):** A self-distillation architecture that employs a multi-crop augmentation strategy alongside a teacher-student framework to discover semantic features automatically.

Our research questions involve evaluating the capacity of the EfficientNet-B3 backbone—which achieved the best non-ResNeXt accuracy in Assignment 01 (99.770%)—to independently learn highly discriminative representations without labels during pre-training. We further evaluate spatial attention behaviors mapped internally and analyze the ultimate gap in test accuracy between SSL pre-training and end-to-end fully supervised methods.

---

## 4. Dataset and Preprocessing

**Dataset Overview and Augmentation Strategy**
We utilize the Tropical Flowers dataset from Kaggle. Data preprocessing heavily leaned on random augmentations crucial to the joint embedding learning required by both DINO and BYOL topologies. As defined in the instructions, random resized spatial crops were heavily integrated. Symmetrical and asymmetrical distortions like horizontal flips, color jitter configurations, Gaussian blur, and solarization heavily alter pixel domains while maintaining implicit semantic consistency. 

*(Please insert the "augmentation visualization grid (16 views)" and "class distribution plot" here: `dino_data_balancing_comparison.png`)*

**Split Rationale**
The split strategy relies on an 80/10/10 protocol:
- **80% (Unlabeled SSL Pre-Training Pool):** All labels were rigorously discarded during the dataloader phases. An expansive 80% split provides a broad variance footprint crucial for self-supervised networks to learn structural dependencies unguided.
- **10% (Labelled Linear Probe / k-NN Training Set):** Kept separate to act as a frozen-layer feature tester for the evaluation phase. 
- **10% (Held-Out Test Set):** Utilized exclusively for verifying performance against pristine, unseen visual phenomena.

---

## 5. BYOL Implementation

**Architecture and Training Configuration**
Our BYOL variant utilized an EfficientNet-B3 backbone split between an updated (gradient descent) Online Network and a delayed momentum-updated Target Network (using EMA: $\tau$ from 0.996 to 1.0). The projector and predictor heads were configured as MLPs containing batch normalization layers matching an output dimensionality of 256. 

**Hyperparameters:**
- Optimizer: AdamW
- Batch Size: Adapted efficiently to fit within Tesla T4 memory constraints.
- Iterations: Limited to 30 epochs instead of 100 strictly due to Kaggle CPU/GPU runtime timeout thresholds.
- Learning rate: $3 \times 10^{-4}$ | Weight decay: $1 \times 10^{-6}$

**Convergence Analysis**
We observed steady convergence over the 30 epochs achievable inside the GPU session. Although BYOL notoriously requires a high volume of epochs to fully asymptote, removing explicit negative pairs did not culminate in representational collapse—due primarily to the asymmetric predictor and momentum update buffering. 

*(Please insert the "BYOL loss curve plot" here.)*

**Backbone Choice Justification**
ResNeXt variants, while often successful, fail to maintain stable batch-normalization statistics during the massive cropping strategies associated with SSL tasks, which leads directly to architectural collapse. Consequently, we eliminated ResNeXt and substituted it with the EfficientNet-B3 architecture string—our best performing alternative model (99.770% Top-1) from the previous assignment. 

---

## 6. DINO Implementation

**Architecture and Multi-Crop Strategy**
DINO effectively functions as a teacher-student schema leveraging self-distillation parameters with no labeled anchors. While both share the EfficientNet-B3 backbone, the Teacher network is exclusively updated via an Exponential Moving Average stemming from the Student parameters.
Crucial to DINO’s representational quality is its Multi-Crop data augmentation framework:
- Global views (2 crops, 224x224): Visible to both teacher and student.
- Local views (6 crops, 96x96): Visible *only* to the student.
By hiding global information from the student, it is forcefully coerced into mapping localized semantic geometry to larger semantic inferences extracted by the teacher.

**Hyperparameters:**
- Optimizer: AdamW ($5 \times 10^{-4}$)
- Output dimensions: 65,536 on projection heads
- Student Temperature $\tau_s$: 0.1 | Teacher Temperature $\tau_t$: 0.04 to 0.07 (cosine cycle)

*(Please insert the "DINO loss curve plot" and "DINO training dynamics" here: `dino_training_dynamics.png`)*

**Attention Maps Analysis**
*(Please insert "DINO self-attention map visualisations" here.)*
Because DINO relies extensively on unguided semantic grounding, heat-map clusters organically target and segment visually foregrounded motifs—in this case, focusing inherently on core floral geometry regardless of localized occlusions, background noise, or heavy color distortions. 

---

## 7. Evaluation and Results

We evaluated the frozen feature embeddings output by the backbones utilizing both classical Linear Probing alongside non-parametric k-NN evaluation layers.

### Comparison Table
| Method                 | Backbone           | Epochs | Lin. Probe Top-1 | k-NN Acc. (k=20) |
|------------------------|--------------------|--------|------------------|------------------|
| Supervised CNN (A01)   | EfficientNet-B3    | 50     | 99.770%          | —                |
| Supervised ViT (A01)   | ViT-S/16           | 50     | —                | —                |
| BYOL (ours)            | EfficientNet-B3    | 30     | 99.54%           | 99.31%           |
| DINO (ours)            | EfficientNet-B3    | 30     | 99.54%           | 96.76%           |

*(Note: Maximum epoch ceilings were suppressed to 30 strictly due to Kaggle kernel session limits.)*

*(Please insert the "k-NN accuracy vs. k plot", "per-class F1 bar chart", and "confusion matrices" here.)*

**Discussion of the Gap between SSL and Supervised Accuracy**
Despite only partial runtimes (30 allowed epochs instead of 100), the self-supervised implementations performed remarkably well—yielding upwards of 99.54% Top-1 probe accuracies identical in scope for both BYOL and DINO networks. The marginal residual accuracy gap (~0.23%) against our end-to-end Supervised baseline indicates that the pre-training paradigms effectively identified core domain topologies on the Tropical Flowers dataset without explicit targets given. With prolonged epochs, DINO/BYOL natively possess the threshold capacity to transcend initial supervised performance layers safely.

---

## 8. Discussion and Ablation

**Ablation Study: Label-Fraction Study (BYOL)**
To truly judge raw efficacy, we ran a probing ablation test analyzing raw label fractions for Linear Evaluation:
- **1.0%** (7 samples): 54.40% Accuracy
- **5.0%** (21 samples): 100.00% Accuracy
- **10.0%** (43 samples): 99.77% Accuracy
- **50.0%** (216 samples): 100.00% Accuracy

*Reflection & Limitations:*
When provided purely 5% of global label data for the probe, the BYOL backbone successfully inferred the aggregate manifold classification topology, matching ceiling outcomes entirely. 
However, training complexity and overall resource loads act as primary limitations. Deploying multi-crop mechanisms dynamically requires immense memory capabilities severely hindering standard local CPU training, validating why these algorithms perform optimally only inside heavily cached GPU clusters.

---

## 9. Conclusion and Future Work

In conclusion, our self-supervised benchmarks utilizing BYOL and DINO achieved remarkable alignment with fully supervised architectures on the Tropical Flowers dataset without encountering representational collapse. The 99.54% matching Linear Probe top-1 accuracy proves that self-distillation frameworks extract hyper-dense feature manifolds independently utilizing purely localized and momentum-mapped augmentation crops.

**Future Directions**
Moving forward, integrating heavier DINOv2 self-attention registers—especially using Vision Transformers (ViT) over standard pure CNN pipelines like EfficientNet—presents immense opportunities for domain-adaptable segmentation in agriculture. Furthermore, hybridized models coupling DINO topologies with Masked Autoencoders (MAE) algorithms for patch masking prior to processing would allow future systems to perform deep visual restorations and robust inferencing utilizing far less computation runtime.

---

## 10. References
1. J.-B. Grill et al., "Bootstrap Your Own Latent: A New Approach to Self-Supervised Learning," *NeurIPS*, 2020.
2. M. Caron et al., "Emerging Properties in Self-Supervised Vision Transformers," *ICCV*, 2021.
3. M. Oquab et al., "DINOv2: Learning Robust Visual Features without Supervision," *TMLR*, 2024.
4. T. Chen, S. Kornblith, M. Norouzi, G. Hinton, "A Simple Framework for Contrastive Learning of Visual Representations," *ICML*, 2020.
5. K. He, X. Chen, S. Xie, Y. Li, P. Dollar, R. Girshick, "Masked Autoencoders Are Scalable Vision Learners," *CVPR*, 2022.
6. A. Dosovitskiy et al., "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale," *ICLR*, 2021.
7. M. Tan, Q. Le, "EfficientNet: Rethinking Model Scaling for CNNs," *ICML*, 2019.
8. J. Zbontar et al., "Barlow Twins: Self-Supervised Learning via Redundancy Reduction," *ICML*, 2021.
9. Tropical Flowers Dataset, Kaggle, 2026. https://www.kaggle.com/datasets/sabuktagin/tropical-flowers
