# Melanoma Detection – Technical Presentation (Markdown Slides)

---

## 10.1) Grad-CAM Basics (Fundamentals)

- Idea: Use gradients of the target w.r.t. feature maps to weight activations and produce a heatmap.
- Interpretation: Highlights regions that contributed most to features for the chosen layer.
- Layer choice:
  - Early layers: finer spatial detail, lower semantics.
  - Late layers: coarser spatial map, higher-level semantics.
- In app: Select layer; we use EigenCAM variant from `pytorch-grad-cam`.

---

## 1) Project Overview

- Goal: Assistive melanoma detection from dermoscopy images with patient metadata.
  - Image + metadata fusion improves performance over image-only.
- Outputs: Probability of malignancy + Grad-CAM heatmaps.
  - Heatmaps help users understand which regions influenced the model.
- Scope: Non-diagnostic; designed for research/education and triage support.
  - Supports reproducible experiments and explainability.

---

## 2) Why This Matters

- Melanoma is lethal when missed; early detection improves outcomes.
  - Sensitivity (recall) is critical to reduce false negatives.
- Assist clinicians with prioritization and visual explanations.
  - Highlighting suspicious regions builds user trust and aids review.
- Standardized pipeline enables reproducible model comparisons.
  - Consistent preprocessing, metrics, and logging across backbones.

---

## 3) Modular Architecture (Refactor Highlights)

- Training: PyTorch Lightning (PL) + TorchMetrics
  - `training/pl_train.py`, `training/pl_module.py`, `training/pl_data.py`
- Model: Image backbone (timm) + Metadata MLP fusion
  - `models/model.py`
- Config: Pydantic BaseSettings + module-level constants
  - `configs/settings.py`, `configs/config.py`
- Data: `timm.data.create_transform` for train/val
  - `data/dataset.py`
- Eval + Explainability: ROC/CM utilities + Grad-CAM (EigenCAM)
  - `eval/evaluate.py`, `app/app.py`
  - TorchMetrics for accuracy, recall, F1; optional TTA.

---

## 4) Data & Features

- Image: RGB dermoscopy preprocessed to `IMAGE_SIZE` (default 256).
- Metadata: `age_approx`, `sex`, `anatom_site_general_challenge`.
  
- Augmentations: Provided by `timm` transforms (train/val consistency).
  - Reduces overfitting; preserves lesion characteristics.
- TTA: Optional at eval; PIL-based in app.
  - Increases robustness by averaging predictions over simple flips/rotations.

---

## 4.1) Classification Task & Classes

- Task: Binary classification
  - Positive class (label=1): Melanoma (malignant)
  - Negative class (label=0): Benign/non-melanoma
- Thresholding
  - Default decision threshold at 0.5 on sigmoid probability
  - Can tune threshold to increase recall (sensitivity) for screening
- Class imbalance
  - Handled via FocalLoss; supports α (class weight) and γ (focusing) from `configs/config.py`
  - Alternative options: weighted sampler, class weights in loss, or oversampling

---

## 5) Model Design (Fusion)

- Image branch: EfficientNet-B0/DenseNet121/ResNet50 backbones (timm, pretrained, num_classes=0).
- Metadata branch: MLP with BN+ReLU and dropout between hidden layers.
- Fusion: Concatenate image + metadata embeddings → Linear head to 1 logit.
  - Keeps architecture simple and efficient while leveraging both modalities.
- Loss: FocalLoss (configurable α, γ, reduction).
  - Mitigates class imbalance by down-weighting easy negatives.

---

## 5.1) Focal Loss (Fundamentals)

- What: Modified cross-entropy that down-weights easy examples and focuses on hard ones.
- Why: Helps with class imbalance and hard-positive mining common in medical imaging.
- How:
  - Focusing parameter γ (>0) reduces loss for well-classified samples.
  - Class weight α ∈ [0,1] rebalances minority/majority classes.
  - Both are configurable in `configs/config.py`.

---

## 6) Training Setup (Lightning)

- Optimizer: Adam, LR from `configs/config.py`.
- Metrics: BinaryAccuracy, BinaryRecall, BinaryF1 (TorchMetrics).
  - Computed on probabilities with 0.5 threshold; monitor `val/f1`.
- Checkpointing: best `val/f1` via ModelCheckpoint.
- Logging: Weights & Biases (project: melanoma-classification).

---

## 6.1) Sigmoid, Logit, and Thresholding (Fundamentals)

- Logit: Raw model output (unbounded real number).
- Sigmoid: Maps logit → probability p ∈ (0,1): `p = 1 / (1 + e^-logit)`.
- Decision rule: Positive if `p ≥ threshold` (default 0.5).
- Tuning threshold: Increase recall by lowering threshold; evaluate precision-recall trade-offs.

---

## 7) Experiments & Metrics (Validation)

Source: `base/*/metrics`

- EfficientNet-B0
  - Acc 97.24% | Recall 88.64% | F1 89.69% | Loss 0.2757
- DenseNet-121
  - Acc 97.10% | Recall 88.93% | Loss 0.2401 (F1 comparable to B0)
- ResNet-50
  - Acc 96.25% | Recall 90.01% | F1 86.69% | Loss 0.2757

Notes:
- Train metrics ~99% across models suggest potential overfitting; regularization and threshold tuning remain important.
- Consider per-model threshold optimization to match clinical sensitivity targets.

---

## 7.1) Metric Definitions (for clarity)

- Accuracy: Fraction of correct predictions over all cases. Can be misleading with imbalance.
- Recall (Sensitivity): Fraction of melanomas correctly identified. High recall = fewer missed cancers.
- Precision: Fraction of positive predictions that are correct. Higher precision = fewer false alarms.
- F1 Score: Harmonic mean of precision and recall. Balances missed cancers vs. false alarms.
- Loss: Optimization objective (BCE/Focal). Lower is better but interpret with F1/recall.

---

## 8) Trade-offs Across Backbones

- EfficientNet-B0
  - Pros: Best F1/accuracy balance, low latency, small footprint.
  - Cons: Slightly lower recall than ResNet-50 in this snapshot.
  - When to use: Default deployment; resource-constrained environments; fast iteration.
- DenseNet-121
  - Pros: Competitive loss; strong representation in medical imaging.
  - Cons: Heavier; marginal improvements vs B0 not evident.
  - When to use: If dataset-specific gains emerge; otherwise B0 is simpler.
- ResNet-50
  - Pros: Highest recall (fewer false negatives).
  - Cons: Lower F1/accuracy; heavier; may need more tuning.
  - When to use: Sensitivity-first screening with threshold tuning/class weighting.

---

## 9) Final Model Choice

- Default: EfficientNet-B0 + Metadata MLP
  - Rationale: Best overall F1/accuracy with efficient runtime and simple deployment.
  - Practical: Stable pretrained weights; CPU-friendly for Docker + Gradio.
- Alternate (sensitivity-first): ResNet-50 + threshold tuning/class weighting.
  
---

## 10) Inference & Explainability

- App: `app/app.py` (Gradio)
  - Single-image upload + metadata inputs.
  - Probability output, Grad-CAM heatmaps, side-by-side visualization.
- Explainability: EigenCAM via `pytorch-grad-cam`.
  - Layer selection influences spatial detail vs. semantics.

---

## 11) Configuration & Reproducibility

- `configs/settings.py`: Pydantic settings, .env support.
- `configs/config.py`: Backward-compatible constants.
- Determinism flag in Lightning trainer; seeds in dataset split.
  - Environment-variable overrides via Pydantic settings with `.env` support.

---

## 12) Deployment

- Dockerfile: CPU-only image; compose service `web` on port 7860.
- Volumes: `./data`, `./result`, `./base` mounted for persistence.
- Healthcheck via `curl` to Gradio endpoint.

---

## 13) Risks & Mitigations

- Overfitting → Use validation monitoring, augmentations, early stopping, cross-validation.
- Spurious correlations (artifacts like rulers) → Inspect Grad-CAM, data curation.
- Class imbalance → FocalLoss, threshold tuning, sampling strategies.
- Domain shift → Validate on target population, consider TTA and robust transforms.

---

## 14) Roadmap

- Cross-validation with confidence intervals.
- Threshold optimization for task-specific recall/precision trade-offs.
- Additional backbones (ConvNeXt-T, EfficientNetV2-S) and self-supervised pretraining.
- ONNX/torchscript export and lightweight serving targets.
- Add `.env` examples and experiment tracking templates.

---

## 15) How to Run (Quick Start)

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

## 16) References

- Timm: https://github.com/huggingface/pytorch-image-models
- PyTorch Lightning: https://www.lightning.ai
- TorchMetrics: https://torchmetrics.readthedocs.io
- Grad-CAM: https://github.com/jacobgil/pytorch-grad-cam
