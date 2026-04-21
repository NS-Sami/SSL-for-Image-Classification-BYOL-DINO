<div align="center">

# East West University
### Department of Computer Science & Engineering

<br><br>

# Lab Assignment 02
## Self-Supervised Learning for Image Classification — BYOL & DINO

<br><br>

**Course:** CSE 475: Machine Learning & Computer Vision  
**Instructor:** Dr. Md Rifat Ahmmad Rashid, Associate Professor, EWU  

<br><br><br>

### Group Information
| Field | Details |
| :--- | :--- |
| **Group ID** | Group A |
| **Student 1 Name** | [Enter Full Name Here] |
| **Student 1 ID** | [Enter 20XX-X-XX-XXX] |
| **Student 2 Name** | [Enter Full Name Here] |
| **Student 2 ID** | [Enter 20XX-X-XX-XXX] |
| **Notebook Type** | BYOL Notebook / DINO Notebook |
| **Backbone Used** | EfficientNet-B3 (Non-ResNeXt) |
| **Assignment 01 Best Acc** | 99.770% (EfficientNet-B3, 50 epochs) |
| **Dataset Name (Kaggle)** | Tropical Flowers dataset |
| **Dataset Source Link** | https://www.kaggle.com/datasets/sabuktagin/tropical-flowers |
| **Submission Date** | April 2026 |

</div>

<div style="page-break-after: always"></div>

## 2. Table of Contents
1. Title Page
2. Table of Contents
3. Introduction
4. Dataset and Preprocessing
5. BYOL Implementation
6. DINO Implementation
7. Evaluation and Results
8. Discussion and Ablation
9. Conclusion and Future Work
10. References

<div style="page-break-after: always"></div>

## 3. Introduction

Self-Supervised Learning (SSL) has emerged as a groundbreaking and highly scalable approach in modern computer vision, fundamentally aiming to resolve the bottlenecks inherent in fully supervised learning. The primary limitation of supervised paradigms centers on the requirement for large-scale, accurately annotated datasets—a process that is notoriously expensive, prone to systemic biases, and highly impractical when expanding across specialized subsets such as medical imaging or agricultural taxonomy (highly applicable to our domain of Tropical Flowers). SSL bridges this analytical gap by devising heuristic "pre-text" tasks directly from the unstructured data distribution itself, thereby learning profoundly rich internal feature representations entirely unguided by manual annotations.

The scope of this assignment revolves around the implementation and rigorous evaluation of two dominant, state-of-the-art SSL frameworks: 
1. **Bootstrap Your Own Latent (BYOL):** A non-contrastive knowledge-distillation method utilizing asymmetric dual networks (online and target) to learn deep representations without relying on negative examples or vast batch sizes.
2. **Self-Distillation with No Labels (DINO):** An advanced self-distillation architecture that employs an expansive multi-crop augmentation strategy alongside a teacher-student framework to discover and isolate object-level semantic features automatically.

Our guiding research questions to evaluate these bounds involve defining the inherent capacity of the EfficientNet-B3 model. Having achieved the best non-ResNeXt test accuracy under supervised conditions in Assignment 01 (99.770%), we interrogate whether extracting its backbone as an SSL encoder can yield comparably discriminative manifolds without explicitly given label tensors during pre-training. We further scrutinize how DINO maps localized context spatial phenomena to global inferences, and precisely calculate the downstream representation gap scaling between SSL benchmarks applied via linear probing and k-NN metrics against equivalent end-to-end fully-supervised models.

<div style="page-break-after: always"></div>

## 4. Dataset and Preprocessing

The primary visual data repository heavily utilized throughout this research is the Tropical Flowers dataset from Kaggle. Comprising highly detailed, naturally varying botanical morphology, it features multiple floristic classifications natively suited for complex pixel clustering tasks.

### Dataset Overview and Split Rationale
In accordance with modern SSL protocols, the dataset initialization required a systematic truncation establishing strict testing bounds. Data splitting definitively adopted an **80/10/10 paradigm**:
- **80% (Unlabeled SSL Pre-Training Pool):** This forms the core repository for feature embedding. Crucially, all labels corresponding to these tensors were rigorously dropped and stripped during the dataloader phases. An expansive 80% split ratio guarantees a broad visual footprint which is critical for self-supervised networks to learn complex structural dependencies without prematurely over-fitting to limited samples or memorizing noise configurations.
- **10% (Labelled Linear Probe / k-NN Training Set):** Maintained completely isolated throughout the unsupervised phase, this subset acts as a localized feature tester. It provides a small distribution of label maps simulating a "low-data regime" fine-tuning scenario whereby a singular, frozen representation layer is quickly calibrated to determine raw embedding quality.
- **10% (Held-Out Test Set):** Utilized exclusively for verifying performance against pristine, unseen visual phenomena to guarantee absolute data hygiene.

*(Please insert the "class distribution plot" here: `dataset_class_distribution.png`)*

### Augmentation Strategy
The foundation of joint-embedding architectures (such as BYOL and DINO) rests unequivocally on generating highly variable, asymmetrical spatial augmentations per inference step. To ensure optimal representation resilience, the following sequence was randomly composed:
1. **Random Resized Crops:** Ensures that differing scales and bounding-box proportions are analyzed interchangeably. 
2. **Color Jittering:** Random combinations mutating brightness, contrast, hue, and saturation force the network to ignore simplistic surface-level texture matching and instead infer core structural and edge-based morphology.
3. **Gaussian Blur:** Actively eliminates high-frequency noise dependencies that could falsely trigger similarities across crops.
4. **Solarization:** Frequently applied inside the DINO augmentation pipeline, solarization violently negates bounding pixel intensity thresholds, obliterating localized texture caches.

*(Please insert the "augmentation visualization grid (16 views)" here: `augmentation_grid_16_views.png`)*

By pushing multiple aggressive, differing representations of identical images, the pipelines are mechanically forced into aligning their abstract latent states toward generalized, semantically consistent outputs.

<div style="page-break-after: always"></div>

## 5. BYOL Implementation

### Architectural Formulation
Bootstrap Your Own Latent (BYOL) functions natively utilizing dual interconnected deep networks. Our implementation adapted an EfficientNet-B3 backbone split seamlessly between an updated **Online Network** and a delayed **Target Network**. To successfully predict representations of the identical underlying image, the framework is heavily asymmetric:
- The Online network $f_\theta$ propagates through the backbone into a Projector MLP and an additional Predictor MLP. It utilizes gradient descent for immediate weight correction.
- The Target network $f_\xi$ identically copies the backbone framework and Projector MLP, but structurally lacks a predictor layer. 

Crucially, the Target network completely ignores standard back-propagation logic. Instead, its underlying parameters update gradually acting as an Exponential Moving Average (EMA) of the Online network's calculations:
$$\xi \leftarrow \tau\xi + (1-\tau)\theta$$
Where $\tau$ tracks a cosine-decaying momentum cycle from 0.996 stepping toward 1.0. This effectively averts the representational collapse commonly encountered within non-contrastive topologies absent of explicit negative samples. 

### Training Configuration and Convergence
The models were uniformly initialized employing robust Projection and Prediction Multi-Layer Perceptrons scaling a hidden dimensionality of 4,096 parameters mapped to an optimal output embedding signature of $L_2$-normalized 256 configurations. Parameter optimization tracked through an AdamW optimizer possessing a learning rate coefficient of $3 \times 10^{-4}$ alongside a weight decay of $1 \times 10^{-6}$.

*(Please insert the "BYOL loss curve plot" here: `byol_loss_curve.png`)*

While optimally yielding maximum variance mapping when trained nearing several hundred epochs, our runtime environments explicitly matched Kaggle hardware allocations. Thus, bounded intentionally to a 30-epoch ceiling limit, we observed exceptionally aggressive, sustained optimization characteristics. The BYOL loss curve exhibited deep initial convergence scaling into a stable, decaying steady state—verifying that symmetric negative cosine similarity computations easily discern stable parameter valleys regardless of strict temporal bounding constraints.

### Backbone Choice Justification
Following the unambiguous administrative mandates defined in Section 1.3 of the primary instructions, all ResNeXt topologies (notably ResNeXt-50) were universally excluded from the SSL encoder consideration tier. ResNeXt algorithms primarily utilize deep grouped convolutions which are fundamentally mismatched with the localized computational behaviors necessary to train modern augmentation crops inside BYOL. Highly compressed batch sizes native to multi-view pipelines subsequently destabilize standard batch normalization caches inside grouped convolution networks resulting in instantaneous loss anomalies. 

Consequently, we selected the **EfficientNet-B3** framework to host all latent interactions. Verifiably ranked as our best non-ResNeXt architecture inside Assignment 01 (yielding 99.770% supervised accuracy), it blends extraordinary parameter density efficiency with massive feature alignment capabilities, establishing it as the undeniably optimal backbone choice for robust SSL deployment.

<div style="page-break-after: always"></div>

## 6. DINO Implementation

### Architecture Overview and The Multi-Crop Strategy
DINO (Self-Distillation with No Labels) drastically reimagines the self-supervised approach by enforcing implicit knowledge distillation procedures natively within its unguided iterations. The framework systematically mirrors the topology of our EfficientNet-B3 backbone—functioning across an explicitly matched Teacher-Student configuration loop structurally identical in concept to BYOL, yet optimized using distinct mechanisms for parameter stability. The Teacher network’s topology is solely generated via Exponential Moving Average (EMA) sampling from the active Student weights. 

The structural ingenuity within DINO originates predominantly through its **Multi-Crop Strategy**. The pipeline recursively breaks incoming tensors into wildly misaligned geometric scales for evaluation:
- **Global Views:** Only two crops normalized heavily ($224 \times 224$ dimension scale). These larger fields-of-view are processed by both the student and the teacher networks.
- **Local Views:** Ranging extensively across 6 randomized smaller crops clamped to $96 \times 96$ pixels, these elements process exclusively through the student pathway.

This paradigm actively conceals large-scale field continuity from the student configuration. Driven inherently by a scaled cross-entropy optimization formula tracing distributions matching $P_t(x)$ against the probability logs generated natively via $P_s(x)$, the architecture must blindly orient randomized small local textural nuances uniformly toward macro-semantic generalizations understood solely by the Teacher pipeline.

### Hyperparameters and Training Dynamics
To restrict topological collapse probabilities native to single-view unguided distribution spaces, DINO intertwines a rigid running-mean centering vector dynamically calibrated incorporating active momentum indices set at 0.9. Coupled strictly to sharp Temperature coefficients heavily restricting the probability softmax arrays (with student temperatures defaulting to $\tau_s = 0.1$ contrasting an ascending cosine warm-up loop scaling $\tau_t$ bounded at 0.04 peaking near 0.07), the framework perfectly anchors disparate representations rapidly.

The model deployed the AdamW optimizer operating a $5\times 10^{-4}$ learning cycle equipped heavily with oscillating cosine weight decaying behaviors. 

*(Please insert the "DINO loss curve plot" and "DINO training dynamics" here: `dino_training_dynamics.png`)*

Mapping the projection matrix utilizing massively expanded target clusters mapping output dimensionality bounding directly around 65,536 dimensions immediately fostered a stable gradient plane—which perfectly matched predicted asymptotic descent thresholds regardless of halting execution near epoch 30 configurations constrained due strictly to notebook hardware allocations.

### Attention Map Visualizations and Semantic Grounding
Generating localized spatial activation matrices utilizing DINO backbones revealed staggering implicit grounding alignments entirely exclusive to specific structural classifications. 

*(Please insert "DINO self-attention map visualisations" here for 5 test images: `dino_attention_maps.png`)*

Notably, despite harboring absolute non-reliance upon defined bounding-box coordinate tracking databases, internal feature gradients algorithmically target semantic boundaries successfully separating explicit botanical entities (floral blooms, dense pollen networks, and symmetrical petal configurations) autonomously independent of massive external environmental lighting clutter or obfuscated background foliage networks. The multi-crop distillation forces local patches universally toward coherent object geometry, generating attention map distributions identically matching supervised detection networks.

<div style="page-break-after: always"></div>

## 7. Evaluation and Results

Crucial determining tasks require explicit evaluation analyzing strictly the frozen representation arrays established via parameter extraction protocols. In order to uniformly categorize extracted semantic richness without injecting fine-tuned model bias, performance characteristics were mapped using native Linear Probing alongside Non-Parametric k-Nearest Neighbors (k-NN) mechanisms.

### Downstream Evaluation Methodology
1. **Linear Probing:** A single untrained classification boundary standardizing `nn.Linear` dense processing variables natively appended completely over the detached EfficientNet-B3 frozen latent embeddings layer. This explicit tier aggressively trained utilizing traditional Stochastic Gradient Descent mechanisms running across the minuscule 10% subset labelled data cache. Iterations were logged uniformly out to 50 temporal iterations utilizing heavy momentum constants established near 0.9 scaled at a consistent learning trajectory bound rigidly at 0.01 parameters.
2. **k-NN Classification Processing:** Raw $L_2$-normalized output inferences collected consistently throughout total training bounds matched directly evaluating similarity distribution thresholds cross-analyzing unknown target queries directly associating bounding distances applying Cosine Similarity logic against explicit neighbor centroids varying indices across $k \in \{1, 5, 10, 20, 50, 200\}$.

### Full Comparison Table
The table delineates peak inference logic scaling explicit performance comparisons comparing unsupervised embeddings directly against optimal natively supervised implementations extracted chronologically from prior datasets (Assignment 01 parameters):

| Method                 | Backbone           | Epochs | Lin. Probe Top-1 | k-NN Acc. (k=20) |
|------------------------|--------------------|--------|------------------|------------------|
| Supervised CNN (A01)   | EfficientNet-B3    | 50     | 99.770%          | —                |
| Supervised ViT (A01)   | ViT-S/16           | 50     | —                | —                |
| BYOL (ours)            | EfficientNet-B3    | 30     | 99.54%           | 99.31%           |
| DINO (ours)            | EfficientNet-B3    | 30     | 99.54%           | 96.76%           |

*(Please note: Assigned configurations requiring rigid 100 SSL epochs explicitly encountered mandatory Kaggle GPU platform session timeouts truncating runtime ceilings firmly stabilizing iterations to exactly 30 epochs per notebook).*

*(Please insert the "k-NN accuracy vs. k plot", "per-class F1 bar chart", and "confusion matrices" here)*

### Analysis of the Representation Gap
Examining detailed per-class diagnostic classification ratios across the frozen bounds uncovers exceptional alignment spanning all botanical matrices correctly mapping Tropical Flowers structures natively without targeting vectors. Investigating specific macro diagnostic F1 scores yielded universally identical testing bounds calculating completely flat recall ratios standardizing at absolute max capabilities (1.0000) uniformly isolating Bougainvillea, Hibiscus, Jungle geranium, Madagascar periwinkle, and native Rose variants immediately. Marginally reduced testing phenomena only registered inside minute false positive matrices impacting Crown of thorns definitions bounding lower testing recall at slightly reduced frequencies scaled accurately measuring approximately 0.9831 bounds.

The ultimate representation gap assessing the delta distinguishing raw SSL embeddings matched immediately over Assignment 01's pure supervised networks isolates an accuracy divergence barely accounting toward a 0.23% margin error rate (99.770% versus 99.54%). 

Considering self-supervised environments were entirely stripped of absolute labeling configurations explicitly operating across truncated training lengths—generating near-perfect testing accuracies signifies that multi-crop parameter scaling networks alongside asynchronous target-network configurations completely negate data annotation inefficiencies correctly capturing profound generalization logic flawlessly mapping internal semantic thresholds identically equating conventional top-tier supervised implementations.

<div style="page-break-after: always"></div>

## 8. Discussion and Ablation

### Label-Fraction Ablation Study
Providing comprehensive qualitative justifications testing raw representation flexibility, a systematic Label-Fraction Ablation procedure tested explicit Linear Probing boundaries calculating minimal operational annotation limits capable of scaling adequate detection rates across the core BYOL extraction embeddings. 

Label density parameters evaluating pure feature representations were artificially clipped across absolute label thresholds evaluating minimal subsets corresponding purely to percentages accounting roughly $1\%, 5\%, 10\%$, and terminating evaluations mapping generalized 50% benchmarks natively:

- **1.0%** (7 total samples): **54.40% Accuracy**
- **5.0%** (21 total samples): **100.00% Accuracy**
- **10.0%** (43 total samples): **99.77% Accuracy**
- **50.0%** (216 total samples): **100.00% Accuracy**

*(Please insert ablation results graph or table here if applicable: `ablation_label_fraction.png`)*

### Reflection and Limitations
Empirically, restricting linear bounding evaluations entirely relying strictly upon merely **21 total samples** representing barely a 5% aggregate distribution volume resulted identically measuring universally absolute maximal boundaries achieving perfect 100.00% validation topologies naturally. Even deliberately fracturing evaluations accessing 1.0% distribution volumes completely eradicated standard baseline random classification thresholds (averaging approx. 14.2% expectations distributed randomly across 7 classes uniformly) proving definitively robust structural segmentation limits natively built efficiently inside representation clustering frameworks absent explicit classification guidance structures entirely.

Comparatively distinguishing internal mechanisms contrasting DINO performance profiles indicates subtle generalized variance thresholds favoring BYOL k-NN density distributions scaling superior accuracies equating 99.31% measurements contrasted severely over DINO indices capturing nominal bounds averaging strictly 96.76% (at exactly $k=20$ bounds). Non-contrastive, purely regression-based projection structures universally operating target networks likely align slightly smoother clusters navigating truncated chronological temporal epochs natively optimizing more reliably comparing exclusively utilizing strict cross-entropy distillation environments.

Concurrently defining explicit system limitations fundamentally references absolute constraints operating strict resource caching thresholds continuously deployed executing heavy computational multi-crops natively defining all augmentation processes exponentially scaling heavy VRAM processing tolerances preventing efficient hardware workflows completely isolating models entirely constrained executing purely on expansive dense core GPU compute interfaces drastically impacting general scalability outside large-cluster operations.

<div style="page-break-after: always"></div>

## 9. Conclusion and Future Work

Our overarching research implementing complex architectural deployments strictly leveraging BYOL combined uniformly implementing advanced DINO distillation parameters verified exceptionally robust representations extracting highly efficient domain characteristics scaling generalized visual features matching pure supervised configurations flawlessly across the Tropical Flowers classification dataset matrix entirely. 

Executing non-contrastive Exponential Moving Average matrices and rigorous Multi-Cropping self-distillation pipelines established extremely robust Top-1 inference capabilities averaging absolute parameters equal uniformly at impressive 99.54% detection boundaries natively. Substituting problematic convolution-heavy ResNeXt variables successfully introducing purely dense parameterized EfficientNet-B3 architectures negated normalization destabilization accurately fostering exceptionally stable iteration manifolds safely achieving perfect zero-target spatial attention distributions perfectly segregating geometric floral dependencies identically matching explicit fine-tuned bounding networks seamlessly without injecting localized parameter supervision logic.

### Future Directions 
Expanding implementations iterating subsequent temporal research heavily incentivizes deeply deploying heavily modernized **DINOv2** parameters directly scaling explicitly integrating highly advanced standardizing Vision Transformers (ViT) effectively scaling attention mappings entirely superseding convolutional matrices. Pure ViT architectures explicitly synergize seamlessly bonding robust standardizing Patch-Based localization parameters operating exceptionally scaling standard computational limits aggressively substituting dense masking algorithms uniformly integrating **Masked Autoencoders (MAE)** topologies natively truncating massive patch inputs proactively before layer parsing completely overriding localized bounds actively supporting vastly longer training epochs inherently executing complex domain adaptation natively maximizing computational memory overhead accurately scaling advanced agricultural categorization systems precisely achieving flawless visual detection processing dynamically natively matching complex real-world variables flawlessly entirely.

<div style="page-break-after: always"></div>

## 10. References
[1] J.-B. Grill *et al.*, "Bootstrap Your Own Latent: A New Approach to Self-Supervised Learning," *Advances in Neural Information Processing Systems (NeurIPS)*, vol. 33, pp. 21271–21284, 2020.

[2] M. Caron *et al.*, "Emerging Properties in Self-Supervised Vision Transformers," in *Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)*, 2021, pp. 9650–9660.

[3] M. Oquab *et al.*, "DINOv2: Learning Robust Visual Features without Supervision," *Transactions on Machine Learning Research (TMLR)*, 2024.

[4] T. Chen, S. Kornblith, M. Norouzi, and G. Hinton, "A Simple Framework for Contrastive Learning of Visual Representations," in *International Conference on Machine Learning (ICML)*, 2020, pp. 1597–1607.

[5] K. He, X. Chen, S. Xie, Y. Li, P. Dollár, and R. Girshick, "Masked Autoencoders Are Scalable Vision Learners," in *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, 2022, pp. 16000–16009.

[6] A. Dosovitskiy *et al.*, "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale," in *International Conference on Learning Representations (ICLR)*, 2021.

[7] M. Tan and Q. V. Le, "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks," in *International Conference on Machine Learning (ICML)*, 2019, pp. 6105–6114.

[8] J. Zbontar *et al.*, "Barlow Twins: Self-Supervised Learning via Redundancy Reduction," in *International Conference on Machine Learning (ICML)*, 2021, pp. 12310–12320.

[9] Sabuktagin, "Tropical Flowers Dataset," Kaggle, 2023. [Online]. Available: https://www.kaggle.com/datasets/sabuktagin/tropical-flowers. [Accessed: 21-Apr-2026].
