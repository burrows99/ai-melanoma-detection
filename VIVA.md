# Viva Preparation: Melanoma Detection (DL + MLOps)

This document compiles likely viva questions with concise answers and simple diagrams tailored to this repo.

---

## 1) Project Overview

- What problem do you solve?
  - Assistive detection of melanoma from dermoscopy images, optionally using patient metadata.
- What is the output?
  - Binary probability (melanoma vs. benign), with Grad-CAM heatmaps for explainability.
- Is this diagnostic?
  - No. Research/education and triage support only.

---

## 2) Data and Preprocessing

- What modalities are used?
  - Image (RGB dermoscopy) + metadata (age, sex, anatomic site).
- What transforms do you use?
  - `timm.data.create_transform` for train/val with ImageNet normalization; consistent with model backbone.
- Any test-time augmentation (TTA)?
  - Optional flips/rotations during eval and in-app averaging.

Diagram: Minimal preprocessing flow
```mermaid
flowchart LR
    A[Raw image] --> B[Resize to IMAGE_SIZE]
    B --> C[Normalize (ImageNet)]
    C --> D[Tensor]
    M[Metadata] --> E[Encode to numeric]
    D --> F[Model]
    E --> F
```

---

## 3) Model Architecture and Fusion

- What architecture do you use?
  - Image backbone (EfficientNet-B0/DenseNet121/ResNet50 via timm, num_classes=0) + Metadata MLP.
- How do you fuse image and metadata?
  - Concatenate embeddings → linear head to a single logit.
- Why not multiply or attention?
  - Concatenation is simple, stable, and effective; attention can be future work if metadata is richer.

Diagram: Fusion
```mermaid
flowchart TB
    I[Image] -->|Backbone (timm)| IE[Img Embedding]
    MD[Metadata] -->|MLP| ME[Meta Embedding]
    IE --> C[Concat]
    ME --> C
    C --> H[Linear Head]
    H --> O[Logit -> Sigmoid]
```

---

## 4) Losses and Metrics

- Why binary logit + sigmoid?
  - Single-output logit for binary classification; sigmoid maps to probability in (0,1).
- What is Focal Loss?
  - Modified cross-entropy that down-weights easy examples and focuses learning on hard/minority samples using γ (focusing) and α (class weight).
- Why Focal Loss here?
  - Addresses class imbalance (melanoma is rarer) and hard positive mining.
- Which metrics do you track?
  - Accuracy, Recall (Sensitivity), F1 via TorchMetrics; watch `val/f1` for model selection.

Formula (conceptual):
```
FocalLoss = - α * (1 - p_t)^γ * log(p_t),  where p_t is the predicted prob of the true class
```

---

## 5) Training Loop and Lightning

- Why PyTorch Lightning?
  - Reduces boilerplate; standardizes logging, checkpointing, and hardware config.
- What do `pl_module.py` and `pl_data.py` do?
  - Model training/validation steps with metrics; data module wraps loaders.
- How do you select the best model?
  - ModelCheckpoint on highest validation F1.

Diagram: Training pipeline (Lightning)
```mermaid
flowchart LR
    DM[LightningDataModule] --> TR[Trainer]
    LM[LightningModule] --> TR
    TR -->|log| WB[W&B]
    TR -->|checkpoint| CKPT[Best .pth]
```

---

## 6) Explainability (Grad-CAM / EigenCAM)

- What is Grad-CAM?
  - Uses gradients of the target with respect to feature maps to weight activations, producing a heatmap highlighting influential regions.
- What is EigenCAM?
  - Class-agnostic variant using principal components of activations; stable, layer-dependent localization.
- How do users select layers?
  - The app exposes target-layer selection; earlier layers → finer detail, later layers → higher-level semantics.

Diagram: Grad-CAM (concept)
```mermaid
sequenceDiagram
  participant Img as Image
  participant Net as CNN
  participant Map as Feature Maps
  participant Hm as Heatmap
  Img->>Net: Forward
  Net->>Map: Activations
  Net-->>Map: Gradients wrt target
  Map->>Hm: Weighted sum + ReLU
  Hm->>Img: Overlay heatmap
```

---

## 7) Classification Details and Thresholding

- Classes:
  - Positive (1): Melanoma; Negative (0): Benign.
- Threshold:
  - Default 0.5; lower to emphasize recall in screening contexts.
- How to choose the threshold?
  - Optimize F1 or sensitivity target using validation PR curve or Youden's J on ROC.

Diagram: Thresholding effect
```mermaid
flowchart LR
    P[Prob] -->|>=0.5| POS[Predict Melanoma]
    P -->|<0.5| NEG[Predict Benign]
```

---

## 8) Results and Trade-offs

- Why EfficientNet-B0 as default?
  - Best F1/accuracy vs. compute/latency; stable pretrained weights; CPU-friendly deployment.
- When to prefer ResNet-50?
  - Sensitivity-first use-cases; apply threshold tuning and/or class weighting.
- DenseNet-121?
  - Competitive but heavier; gains not evident vs. B0 in our runs.

---

## 9) Evaluation and Overfitting

- Signs of overfitting?
  - Very high train accuracy vs. lower validation metrics.
- Mitigations used?
  - Proper augmentation, monitoring val F1, early stopping/checkpointing, metadata fusion.
- Additional steps?
  - Cross-validation, stronger regularization, and more diverse data.

---

## 10) Deployment and Reproducibility

- How to serve?
  - Gradio app (`app/app.py`), Dockerized; CPU-only image for portability.
- Reproducibility?
  - Pydantic settings with `.env`; fixed seeds; standardized transforms with timm; Lightning trainer flags.
- Monitoring/logging?
  - W&B for metrics; structured checkpoints in `result/weights/`.

---

## 11) Ethics and Risk

- Not diagnostic; must be supervised by clinicians.
- Bias and domain shift risks; validate on target cohorts.
- Explainability aids error analysis but is not ground truth.

---

## 12) Quick Commands to Remember

- Train (Lightning):
```bash
python -m training.pl_train
```
- Evaluate:
```bash
python -m eval.evaluate
```
- Serve App:
```bash
python app/app.py
```

---

## 13) Extra: Confusion Matrix Cheat Sheet

```
             Pred +   Pred -
Actual +       TP        FN
Actual -       FP        TN

Accuracy = (TP+TN)/(TP+TN+FP+FN)
Recall = TP/(TP+FN)
Precision = TP/(TP+FP)
F1 = 2 * (Precision*Recall) / (Precision+Recall)
```
