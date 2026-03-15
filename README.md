# Multi-Image Attention Fusion for Musculoskeletal Abnormality Detection

> Automated detection of musculoskeletal abnormalities in X-ray studies using gated spatial attention pooling, study-level multi-image fusion, and per-category classification heads. Best single model achieves **κ = 0.737** on MURA, exceeding the Stanford DenseNet-169 baseline (κ ≈ 0.705) and approaching reported radiologist-level agreement (κ ≈ 0.77).

---

## Overview

Musculoskeletal conditions account for a significant portion of radiological workload, yet automated diagnosis remains challenging due to the multi-image nature of X-ray studies, large within-category variance, and class imbalance across body regions. This project investigates how large pretrained convolutional and transformer backbones can be adapted for grayscale multi-image radiograph classification, with a focus on study-level reasoning across variable numbers of images.

The core contribution is an architecture that treats each radiological study as a **bag of images** and performs two stages of learned attention: (1) spatial attention pooling within each image to extract a fixed-length representation, and (2) cross-image attention across the study to fuse representations into a single classification decision — both conditioned on the body region category.

---

## Dataset

**MURA v1.1** (Stanford ML Group) — a large-scale musculoskeletal radiograph dataset.

| Split | Studies | Images |
|-------|---------|--------|
| Train | 13,457 | 36,808 |
| Valid | 1,199  | 3,197  |

Seven body region categories: `SHOULDER`, `HUMERUS`, `ELBOW`, `FOREARM`, `WRIST`, `HAND`, `FINGER`.

Each study contains a variable number of images (typically 2–5 views) and a single binary label (normal / abnormal). The dataset exhibits moderate class imbalance (≈40% abnormal overall) that varies significantly across categories.

---

## Architecture

### Overview

```
Study (N images, category c)
        │
        ▼
┌───────────────────┐
│  Shared Backbone  │  ConvNeXt-L / ViT-L/16 / ResNet-152
│  (grayscale)      │
└────────┬──────────┘
         │  spatial feature map (N, C, H, W)
         ▼
┌───────────────────┐
│  Gated Spatial    │  per-image attention pooling
│  Attention Pool   │  → (N, embed_dim)
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│  Cross-Image      │  per-category MHA over study images
│  Attention Fusion │  → (1, embed_dim)
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│  Per-Category     │  separate linear head per body region
│  Classifier       │
└────────┬──────────┘
         │
         ▼
     logit (scalar)
```

### Gated Spatial Attention Pooling

Standard global average pooling discards spatial structure that is critical for localised pathology detection. Instead, each image's feature map is pooled using **gated MIL attention** — a two-gate mechanism that independently models what features are present and how important each spatial location is:

```
spatial  = flatten(backbone(x))          # (N, H×W, C)
scores   = W_w · (tanh(W_v · spatial) ⊙ σ(W_u · spatial))   # (N, H×W, 1)
weights  = softmax(scores)
pooled   = Σ weights_i · spatial_i       # (N, C)
```

The tanh gate encodes feature content; the sigmoid gate acts as a differentiable binary mask. Their elementwise product sharpens attention to regions where both gates agree, suppressing uninformative background. This is strictly more expressive than single-gate attention while remaining computationally negligible relative to the backbone.

### Study-Level Cross-Image Fusion

A radiological study consists of multiple views of the same anatomy. Simple averaging across images loses the inter-view relationships that radiologists exploit (e.g. confirming a finding across AP and lateral views). Instead, the pooled per-image embeddings are fused using **multi-head self-attention**:

```
seq      = stack([embed_1, ..., embed_N])     # (1, N, embed_dim)
attended = MHA_c(seq, seq, seq)               # per-category attention
weights  = softmax(W_c · attended)            # soft image weighting
fused    = Σ weights_i · embed_i             # (1, embed_dim)
logit    = classifier_c(fused)
```

This allows the model to learn that some views are more diagnostically informative than others, and to aggregate evidence across views in a content-dependent, category-specific way.

### Per-Category Specialisation

All attention, normalisation, weighting, and classification modules are **instantiated separately per body region**:

```python
self.inter_attn  = ModuleDict({cat: MultiheadAttention(...) for cat in categories})
self.inter_ln    = ModuleDict({cat: RMSNorm(embed_dim)      for cat in categories})
self.weight_head = ModuleDict({cat: Linear(embed_dim, 1)    for cat in categories})
self.classifiers = ModuleDict({cat: Sequential(Dropout, Linear(embed_dim, 1)) for cat in categories})
```

This adds ~1.84M parameters but allows each category's fusion strategy to be learned independently — a shoulder study has structurally different diagnostic cues from a finger study, and sharing these heads would conflate them.

### Grayscale Adaptation

Pretrained RGB weights are preserved by averaging the three input channels of the stem convolution into a single-channel equivalent:

```python
new_conv.weight.data = old_conv.weight.data.mean(dim=1, keepdim=True)
```

This initialises the grayscale model at a semantically meaningful point in weight space — substantially better than random initialisation — while correctly handling single-channel radiograph inputs.

---

## Training

### Loss

Focal Binary Cross-Entropy with per-sample positive weighting:

```
L = Σ (1 - p_t)^γ · BCE(logit, label, pos_weight)
```

`γ = 1.5` downweights easy correct predictions, focusing capacity on ambiguous and hard cases. Positive weight `pos_weight = 1.47` corrects for class imbalance.

### Progressive Unfreezing

The backbone is initially frozen and progressively thawed from the final stage inward, with newly unfrozen parameter groups receiving a scaled learning rate (`lr_scale = 0.1`). In prescheduled mode, one group is unfrozen every `unfreeze_patience` epochs regardless of metric plateau, ensuring full unfreezing within a fixed budget.

### Learning Rate Schedule

Cosine decay with linear warmup:

```
lr(t) = lr_base × 0.5 × (1 + cos(π × progress))
```

Applied per-step, with each parameter group's effective LR scaled by its `lr_scale` factor.

---

## Results

### Backbone Ablation

All models trained for 50 epochs, `lr=1e-5`, `pos_weight=1.47`, `embed_dim=256`, with identical classifier heads. Resolution is backbone-native pretrained resolution.

| Backbone | Resolution | Params | κ (untuned) | κ (tuned) | Acc |
|----------|-----------|--------|-------------|-----------|-----|
| ResNet-152 | 512×512 | 60M | 0.650 | — | 0.830 |
| ViT-L/16 | 224×224 | 307M | 0.661 | — | 0.835 |
| ConvNeXt-L | 384×384 | 197M | 0.699 | — | 0.853 |
| ConvNeXt-L + TTA | 384×384 | 197M | 0.699 | **0.737** | **0.871** |

The gap between ResNet-152 (512×512) and ConvNeXt-L (384×384) despite lower resolution suggests that architectural modernity (depthwise convolutions, LayerNorm, GELU activations) matters more than resolution at this scale. The gap between ViT-L/16 and ConvNeXt-L at comparable parameter counts suggests that inductive spatial biases in CNNs remain advantageous for this task, likely due to the local nature of radiological findings.

### Per-Category Results (ConvNeXt-L + TTA)

| Category | κ | Acc | Threshold |
|----------|---|-----|-----------|
| SHOULDER | 0.681 | — | 0.21 |
| HUMERUS  | 0.867 | — | 0.59 |
| ELBOW    | 0.764 | — | 0.36 |
| FOREARM  | 0.727 | — | 0.66 |
| WRIST    | 0.805 | — | 0.30 |
| HAND     | 0.577 | — | 0.76 |
| FINGER   | 0.712 | — | 0.42 |
| **Overall** | **0.737** | **0.871** | — |

The wide per-category threshold spread (0.21–0.76) reflects systematic miscalibration across body regions — the model is consistently underconfident for SHOULDER and overconfident for HAND. HAND remains the weakest category, likely due to its high bone count and small bone sizes relative to image resolution.

### Comparison to Baselines

| System | κ |
|--------|---|
| Stanford DenseNet-169 (Rajpurkar et al. 2018) | ≈ 0.705 |
| Radiologist average (reported) | ≈ 0.770 |
| **This work (ConvNeXt-L + TTA)** | **0.737** |

### Test-Time Augmentation

TTA is applied over 7 variants: original, horizontal flip, vertical flip, 90°/180°/270° rotation, and contrast adjustment (×0.85 and ×1.15). Logits are averaged before sigmoid, and per-category thresholds are tuned on the validation set post-hoc.

---

## Backbones Evaluated

| Model | timm ID | Resolution | Backbone Params | Output Channels |
|-------|---------|-----------|-----------------|-----------------|
| ResNet-152 | `resnet152` | 512×512 | 60M | 2048 |
| ViT-L/16 | `vit_l_16` | 224×224 | 307M | 1024 |
| ConvNeXt-L | `convnext_large.fb_in22k_ft_in1k_384` | 384×384 | 197M | 1536 |
| ConvNeXt-XL | `convnext_xlarge.fb_in22k_ft_in1k_384` | 384×384 | 350M | 2048 |
| NFNet-F4 | `dm_nfnet_f4.dm_in1k` | 512×512 | 316M | 3072 |

ConvNeXt-XL produced comparable results to ConvNeXt-L, suggesting resolution is a stronger bottleneck than parameter count at this scale. NFNet-F4 targets this directly with native 512×512 operation and higher channel dimensionality.

---

## Observations and Future Work

- **Calibration is the primary remaining challenge.** Per-category threshold tuning adds ~0.038 κ, which is a large fraction of the gap to radiologist performance. Temperature scaling or label smoothing during training may reduce this gap without post-hoc tuning.

- **HAND is a persistent weak point.** Small bone pathology at 384×384 may require either higher resolution inputs or a specialised augmentation strategy for this category.

- **Resolution vs. architecture capacity.** The ablation suggests resolution gains plateau before parameter gains — ConvNeXt-L → XL (same resolution, +153M params) had minimal impact, while the ConvNeXt family consistently outperformed ResNet-152 at lower resolution but higher architectural modernity.

- **Ensemble potential.** Single model at κ=0.737; a full ensemble of ConvNeXt-L + ViT-L/16 + NFNet-F4 with TTA and per-category tuning is the most direct path to radiologist-level performance.

- **Domain pretraining.** All backbones use ImageNet pretraining. Medical-domain pretraining (e.g. CheXpert-pretrained encoders) could reduce the domain gap from natural images to radiographs and may particularly help with HAND and SHOULDER.

---

## Reference

Rajpurkar, P., Irvin, J., Ball, R. L., Zhu, K., Yang, B., Mehta, H., ... & Lungren, M. P. (2018). *Mura: Large dataset for abnormality detection in musculoskeletal radiographs.* arXiv:1712.06957.