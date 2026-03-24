"""
Usage:
    uv run vis_infer.py
    uv run vis_infer.py --checkpoint models/convnext_l/best_model_convnext_l.pt
    uv run vis_infer.py --category XR_WRIST --index 42
"""

import argparse
import random
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from PIL import Image

import architectures.convnext_l
import architectures.vit_l_16
import architectures.resnet152
try:
    import architectures.nfnet_f4
    import architectures.convnext_xl
except ImportError:
    pass

from helpers.checkpoint import load_checkpoint
from helpers.patientdataset import load_df
from helpers.augments import make_transform

# ── Backbone forward: return spatial attention weights ─────────────────────────

def backbone_forward_with_attn(backbone_module, x):
    feats   = backbone_module.backbone(x)                      # (N, C, H, W)
    H, W    = feats.shape[2], feats.shape[3]
    spatial = feats.flatten(2).transpose(1, 2)                 # (N, H*W, C)
    scores  = backbone_module.attn_w(
        torch.tanh(backbone_module.attn_v(spatial)) *
        torch.sigmoid(backbone_module.attn_u(spatial))
    )                                                          # (N, H*W, 1)
    weights = F.softmax(scores, dim=1)                         # (N, H*W, 1)
    pooled  = (weights * spatial).sum(dim=1)                   # (N, C)
    out     = backbone_module.proj(pooled)                     # (N, embed_dim)
    spatial_weights = weights.squeeze(-1).reshape(-1, H, W)    # (N, H, W)
    return out, feats, spatial_weights


def classifier_forward_with_attn(model, image_list, categories):
    logits, cross_weights, spatial_weights_list, feats_list = [], [], [], []

    for images, category in zip(image_list, categories):
        images = (torch.stack(images) if isinstance(images, list) else images)
        images = images.to(next(model.parameters()).device)

        context, feats, sp_w = backbone_forward_with_attn(model.backbone, images)

        seq         = context.unsqueeze(0)
        attended, _ = model.inter_attn[category](seq, seq, seq)
        attended    = model.inter_ln[category](attended.squeeze(0))
        weights     = F.softmax(model.weight_head[category](attended), dim=0)
        fused       = (weights * context).sum(dim=0, keepdim=True)
        logit       = model.classifiers[category](fused).squeeze()

        logits.append(logit)
        cross_weights.append(weights.squeeze(1))
        spatial_weights_list.append(sp_w)
        feats_list.append(feats)

    return torch.stack(logits), cross_weights, spatial_weights_list, feats_list


# ── GradCAM ────────────────────────────────────────────────────────────────────

def compute_gradcam(model, category, feats):
    feats_g = feats.detach().requires_grad_(True)
    H, W    = feats_g.shape[2], feats_g.shape[3]

    # re-run the full head from feats forward so grad flows through feats_g
    spatial = feats_g.flatten(2).transpose(1, 2)
    scores  = model.backbone.attn_w(
        torch.tanh(model.backbone.attn_v(spatial)) *
        torch.sigmoid(model.backbone.attn_u(spatial))
    )
    attn_w  = F.softmax(scores, dim=1)
    pooled  = (attn_w * spatial).sum(dim=1)
    context = model.backbone.proj(pooled)

    seq         = context.unsqueeze(0)
    attended, _ = model.inter_attn[category](seq, seq, seq)
    attended    = model.inter_ln[category](attended.squeeze(0))
    weights     = F.softmax(model.weight_head[category](attended), dim=0)
    fused       = (weights * context).sum(dim=0, keepdim=True)
    logit       = model.classifiers[category](fused).squeeze()

    model.zero_grad()
    logit.backward()

    grads = feats_g.grad                                # (N, C, H, W)
    alpha = grads.mean(dim=(2, 3), keepdim=True)        # (N, C, 1, 1)
    cam   = F.relu((alpha * feats_g.detach()).sum(dim=1))  # (N, H, W)

    out = []
    for c in cam:
        mn, mx = c.min(), c.max()
        out.append(((c - mn) / (mx - mn + 1e-8)).cpu())
    return torch.stack(out)                             # (N, H, W) cpu


# ── Rendering ─────────────────────────────────────────────────────────────────

def _upsample_np(heatmap_tensor, h, w):
    t = heatmap_tensor.unsqueeze(0).unsqueeze(0).float()
    t = F.interpolate(t, size=(h, w), mode="bilinear", align_corners=False)
    return t.squeeze().numpy()


def _styled_ax(ax, title, border_color, border_lw):
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=8, color="#e2e8f0", pad=3)
    for spine in ax.spines.values():
        spine.set_edgecolor(border_color)
        spine.set_linewidth(border_lw)


def render_image_pair(ax_attn, ax_grad, img_tensor, spatial_w, gradcam_w,
                      cross_w_scalar, img_idx, n_images):
    img_np = img_tensor.squeeze(0).numpy()
    img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)
    h, w   = img_np.shape

    attn_up = _upsample_np(spatial_w, h, w)
    attn_up = (attn_up - attn_up.min()) / (attn_up.max() - attn_up.min() + 1e-8)

    grad_up = _upsample_np(gradcam_w, h, w)
    grad_up = (grad_up - grad_up.min()) / (grad_up.max() - grad_up.min() + 1e-8)

    border_color = plt.cm.YlOrRd(cross_w_scalar)
    lw           = 4 * cross_w_scalar * n_images + 1

    ax_attn.imshow(img_np, cmap="gray", vmin=0, vmax=1)
    ax_attn.imshow(attn_up, cmap="inferno", alpha=0.45, vmin=0, vmax=1)
    _styled_ax(ax_attn, f"Img {img_idx+1}  α={cross_w_scalar:.3f}\nMIL Attention",
               border_color, lw)

    ax_grad.imshow(img_np, cmap="gray", vmin=0, vmax=1)
    ax_grad.imshow(grad_up, cmap="viridis", alpha=0.45, vmin=0, vmax=1)
    _styled_ax(ax_grad, f"Img {img_idx+1}  α={cross_w_scalar:.3f}\nGradCAM",
               border_color, lw)


def visualize_study(images_tensor, spatial_weights, gradcam_weights, cross_weights,
                    prob, label, category, save_path):
    N  = images_tensor.shape[0]
    cw = cross_weights.cpu().numpy()

    studies_per_row = 3
    n_study_rows    = (N + studies_per_row - 1) // studies_per_row
    ncols           = min(N, studies_per_row) * 2
    nrows           = n_study_rows + 1

    fig = plt.figure(figsize=(ncols * 2.8, nrows * 3.2 + 1.5), facecolor="#0f1117")
    correct      = (prob > 0.5) == (label == 1)
    result_color = "#34d399" if correct else "#f87171"
    fig.suptitle(
        f"{category.replace('XR_', '')}  ·  "
        f"Pred: {'Abnormal' if prob > 0.5 else 'Normal'} ({prob:.1%})  ·  "
        f"GT: {'Abnormal' if label == 1 else 'Normal'}  ·  "
        f"{'✓ Correct' if correct else '✗ Wrong'}",
        color=result_color, fontsize=13, fontweight="bold", y=0.99
    )

    gs = gridspec.GridSpec(nrows, ncols, figure=fig, hspace=0.45, wspace=0.08)

    for i in range(N):
        study_row = i // studies_per_row
        col_base  = (i % studies_per_row) * 2
        ax_attn   = fig.add_subplot(gs[study_row, col_base])
        ax_grad   = fig.add_subplot(gs[study_row, col_base + 1])
        render_image_pair(ax_attn, ax_grad,
                          images_tensor[i], spatial_weights[i], gradcam_weights[i],
                          float(cw[i]), i, N)

    # hide unused cells
    used_in_last_row = N % studies_per_row or studies_per_row
    for j in range(used_in_last_row, studies_per_row):
        for c in [j * 2, j * 2 + 1]:
            if c < ncols:   # ← guard
                ax = fig.add_subplot(gs[n_study_rows - 1, c])
                ax.set_visible(False)

    # fusion weight bar chart
    ax_bar = fig.add_subplot(gs[nrows - 1, :])
    ax_bar.set_facecolor("#1e293b")
    bars = ax_bar.bar(range(N), cw, color=[plt.cm.YlOrRd(w) for w in cw],
                      edgecolor="#334155", width=0.5)
    for bar, val in zip(bars, cw):
        ax_bar.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
                    f"{val:.3f}", ha="center", va="bottom", color="#e2e8f0", fontsize=8)
    ax_bar.set_xticks(range(N))
    ax_bar.set_xticklabels([f"Img {i+1}" for i in range(N)], color="#94a3b8", fontsize=8)
    ax_bar.set_ylabel("Fusion Weight α", color="#94a3b8", fontsize=9)
    ax_bar.set_title("Study-Level Cross-Image Fusion Weights", color="#cbd5e1", fontsize=10)
    ax_bar.tick_params(colors="#64748b")
    ax_bar.set_ylim(0, max(cw) * 1.3)
    for spine in ax_bar.spines.values():
        spine.set_edgecolor("#334155")

    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="#0f1117")
    plt.close()
    print(f"Saved → {save_path}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="models/convnext_l/best_model_convnext_l.pt")
    parser.add_argument("--data_dir",   default="data/MURA-v1.1")
    parser.add_argument("--parent_dir", default="data")
    parser.add_argument("--split",      default="valid", choices=["train", "valid"])
    parser.add_argument("--category",   default=None)
    parser.add_argument("--index",      type=int, default=None)
    parser.add_argument("--size",       type=int, default=384)
    parser.add_argument("--out",        default="infer")
    parser.add_argument("--seed",       type=int, default=None)
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading checkpoint: {args.checkpoint}")
    model, _ = load_checkpoint(args.checkpoint, device=device)
    model.eval()

    csv_name = "train_image_paths.csv" if args.split == "train" else "valid_image_paths.csv"
    df       = load_df(csv_name, args.data_dir)

    CATEGORIES = ["XR_SHOULDER", "XR_HUMERUS", "XR_ELBOW",
                  "XR_FOREARM",  "XR_WRIST",   "XR_HAND", "XR_FINGER"]
    category = args.category or random.choice(CATEGORIES)
    cat_df   = df[df["category"] == category].reset_index(drop=True).copy()

    cat_df["studyId"]   = cat_df["image_path"].apply(lambda x: x.split("/")[4])
    cat_df["study_key"] = cat_df["patientId"] + "_" + cat_df["studyId"]
    studies    = cat_df.groupby("study_key")
    study_keys = list(studies.groups.keys())
    idx        = args.index if args.index is not None else random.randint(0, len(study_keys) - 1)
    study_key  = study_keys[idx % len(study_keys)]
    study_df   = studies.get_group(study_key)
    label      = int(study_df["label"].iloc[0])

    print(f"Category: {category}  |  Study: {study_key}  |  "
          f"Label: {'Abnormal' if label else 'Normal'}  |  Images: {len(study_df)}")

    transform = make_transform(augment=False, size=args.size)
    images    = [transform(Image.open(Path(args.parent_dir) / p).convert("L"))
                 for p in study_df["image_path"].tolist()]
    images_stacked = torch.stack(images).to(device)

    # attention pass — no grad needed
    with torch.no_grad():
        logits, cross_weights, spatial_weights, feats_list = classifier_forward_with_attn(
            model, [images_stacked], [category]
        )

    prob    = logits[0].sigmoid().item()
    cross_w = cross_weights[0].cpu()
    sp_w    = spatial_weights[0].cpu()
    feats   = feats_list[0]

    # GradCAM — needs grad, runs separately
    gradcam_w = compute_gradcam(model, category, feats)

    print(f"Prediction: {'Abnormal' if prob > 0.5 else 'Normal'} ({prob:.1%})")
    for i, (cw_i, path) in enumerate(zip(cross_w.tolist(), study_df["image_path"].tolist())):
        print(f"  Image {i+1}: fusion_α={cw_i:.4f}  ({Path(path).name})")

    out_dir   = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    existing  = list(out_dir.glob("viz_*.png"))
    next_n    = max((int(p.stem.split("_")[1]) for p in existing), default=0) + 1
    save_path = out_dir / f"viz_{next_n:04d}_{category.replace('XR_', '').lower()}.png"

    visualize_study(images_stacked.cpu(), sp_w, gradcam_w, cross_w,
                    prob, label, category, save_path)


if __name__ == "__main__":
    main()