import torch
import numpy as np
from torch.amp import autocast
from tqdm.auto import tqdm
from sklearn.metrics import cohen_kappa_score, accuracy_score, confusion_matrix

from architectures.classifier import Classifier
from architectures.convnext_l import ConvNeXt_L_Backbone
from helpers.checkpoint import load_checkpoint
from helpers.patientdataset import load_df, make_loader
from helpers.augments import tta_variants

# ── Config ────────────────────────────────────────────────────────────────────

DEVICE     = "cuda"
DATA_DIR   = "data/MURA-v1.1"
PARENT_DIR = "data"
CATEGORIES = ["XR_SHOULDER", "XR_HUMERUS", "XR_ELBOW",
              "XR_FOREARM",  "XR_WRIST",   "XR_HAND", "XR_FINGER"]

# Each entry: (checkpoint_path, input_size, weight)
# weight is per-model global, for per-category weighting fill PER_CAT_WEIGHTS
MODELS = [
    ("models/best_model_convnext_l.pt",  384, 0.699),
    #("models/best_model_vit_l_16.pt",    224, 0.661),
    #("models/best_model_resnet152.pt",   512, 0.650),
]

# Optional per-category weights (n_models, n_categories), same order as MODELS/CATEGORIES
# Set None to use the global weights from MODELS instead
#PER_CAT_WEIGHTS = [
#    [0.59, 0.82, 0.75, 0.65, 0.78, 0.53, 0.75],  # ConvNeXt-L
#    [0.62, 0.73, 0.73, 0.60, 0.71, 0.57, 0.63],  # ViT-L/16
#    [0.63, 0.84, 0.60, 0.71, 0.72, 0.42, 0.59],  # ResNet-152
#]
PER_CAT_WEIGHTS = None

TTA = False   #test-time augmentation

THRESH_RANGE = np.arange(0.20, 0.81, 0.01)

# ── Inference ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def infer(model, loader):
    logits_out, labels_out, cats_out = [], [], []
    for image_list, labels, categories in tqdm(loader, leave=False):
        with autocast(device_type="cuda"):
            variants = tta_variants(image_list) if TTA else [image_list]
            logits = sum(model(v, categories)[0] for v in variants) / len(variants)
        logits_out.extend(logits.cpu().float().tolist())
        labels_out.extend(labels.cpu().int().tolist())
        cats_out.extend(categories)
    return np.array(logits_out), np.array(labels_out), cats_out

# ── Build ensemble logits ─────────────────────────────────────────────────────

cat_to_idx = {c: i for i, c in enumerate(CATEGORIES)}
all_logits, labels, categories = [], None, None

for ckpt, size, _ in MODELS:
    print(f"Running {ckpt} @ {size}x{size}...")
    model, _ = load_checkpoint(ckpt, device=DEVICE)
    model.eval()
    loader = make_loader(load_df("valid_image_paths.csv", DATA_DIR), augment=False,
                         parent_dir=PARENT_DIR, size=size, batch_size=8,
                         shuffle=False, num_workers=2, pin_memory=True)
    logits, labels, categories = infer(model, loader)
    all_logits.append(logits)

# Normalise weights (global or per-category)
if PER_CAT_WEIGHTS is not None:
    W = np.array(PER_CAT_WEIGHTS)                          # (M, C)
    W = W / W.sum(axis=0, keepdims=True)
    cat_indices = np.array([cat_to_idx[c] for c in categories])
    weights_per_sample = np.stack([W[m][cat_indices] for m in range(len(MODELS))], axis=0)  # (M, N)
else:
    global_w = np.array([w for _, _, w in MODELS])
    global_w /= global_w.sum()
    weights_per_sample = np.tile(global_w[:, None], (1, len(labels)))  # (M, N)

ensemble_logits = (np.stack(all_logits, axis=0) * weights_per_sample).sum(axis=0)
ensemble_probs  = 1 / (1 + np.exp(-ensemble_logits))

# ── Per-category threshold tuning ────────────────────────────────────────────

best_thresh = {}
for cat in CATEGORIES:
    mask = np.array([c == cat for c in categories])
    p, l = ensemble_probs[mask], labels[mask]
    best_t, best_k = 0.5, -1.0
    for t in THRESH_RANGE:
        try:
            k = cohen_kappa_score(l, (p > t).astype(int))
            if k > best_k:
                best_k, best_t = k, t
        except Exception:
            pass
    best_thresh[cat] = best_t

preds = np.array([int(ensemble_probs[i] > best_thresh[categories[i]])
                  for i in range(len(labels))])

# ── Report ────────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print(f"  ENSEMBLE REPORT  ({len(MODELS)} model(s), TTA={'on' if TTA_FLIP else 'off'})")
print("=" * 60)

print(f"\n  Global k (t=0.50): {cohen_kappa_score(labels, (ensemble_probs>0.5).astype(int)):.4f}")
print(f"  Global k (tuned):  {cohen_kappa_score(labels, preds):.4f}")
print(f"  Accuracy (tuned):  {accuracy_score(labels, preds):.4f}")

print(f"\n  {'Category':<10} {'Thresh':>7}  {'k':>7}  {'Acc':>7}  {'Sens':>7}  {'Spec':>7}")
print("  " + "-" * 52)
for cat in CATEGORIES:
    mask = np.array([c == cat for c in categories])
    l    = labels[mask]
    p    = (ensemble_probs[mask] > best_thresh[cat]).astype(int)
    tn, fp, fn, tp = confusion_matrix(l, p).ravel()
    print(f"  {cat.replace('XR_',''):<10}"
          f" {best_thresh[cat]:>7.2f}"
          f" {cohen_kappa_score(l, p):>7.4f}"
          f" {accuracy_score(l, p):>7.4f}"
          f" {tp/(tp+fn) if (tp+fn) else 0:>7.3f}"
          f" {tn/(tn+fp) if (tn+fp) else 0:>7.3f}")

print("=" * 60)