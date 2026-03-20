'''
uv run pretrain.py --msk-only
'''

import argparse, math, random
from pathlib import Path

import torch, torch.nn as nn, torch.nn.functional as F
import timm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.amp import autocast, GradScaler
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from PIL import Image
from tqdm.auto import tqdm

# ── Config ─────────────────────────────────────────────────────────────────────

DATA_DIRS = {
    # MURA: normal only — abnormal images would corrupt the "what should be here" prior
    "mura_normal": ["data/MURA-v1.1"],
    "msk":         ["data/rsna_bone_age", "data/fracatlas", "data/fracatlas_orig",
                    "data/fracture_msk", "data/stanford_bone_age",
                    "data/grazpedwri", "data/knee_oa"],
    "cxr":         ["data/nih_chest_xray14", "data/chexpert"],
}
IMAGE_EXTS = {".png", ".jpg", ".jpeg"}
OUT_DIR    = Path("models/pretrained")

# ── Dataset ────────────────────────────────────────────────────────────────────

class XRayDataset(Dataset):
    def __init__(self, roots, size=384, mura_normal_only=True):
        paths = []
        for root in roots:
            root = Path(root)
            if not root.exists():
                continue
            for p in root.rglob("*"):
                if p.suffix.lower() not in IMAGE_EXTS:
                    continue
                # filter out MURA abnormal studies
                if mura_normal_only and "MURA" in str(p) and "positive" in str(p):
                    continue
                paths.append(p)
        self.paths = paths
        self.transform = T.Compose([
            T.Grayscale(), T.Resize((size, size)), T.ToTensor(),
            T.Normalize(mean=[0.449], std=[0.226])
        ])
        self.aug = T.Compose([
            T.Grayscale(), T.Resize((size, size)),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.2),
            T.RandomAffine(degrees=25, translate=(0.1, 0.1), scale=(0.8, 1.2)),
            T.RandomAdjustSharpness(2, p=0.4),
            T.RandomAutocontrast(p=0.4),
            T.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),
            T.ToTensor(),
            T.Normalize(mean=[0.449], std=[0.226])
        ])
        print(f"  Dataset: {len(self.paths):,} images from {len(roots)} dirs")

    def __len__(self):  return len(self.paths)

    def __getitem__(self, i):
        try:
            img = Image.open(self.paths[i]).convert("L")
            return self.transform(img), self.aug(img)
        except Exception:
            return self[random.randint(0, len(self) - 1)]

# ── Patch masking ──────────────────────────────────────────────────────────────

def mask_patches(x, patch_size=32, mask_ratio=0.75):
    B, C, H, W = x.shape
    ph, pw     = H // patch_size, W // patch_size
    n_patches  = ph * pw
    rand    = torch.rand(B, n_patches, device=x.device)
    mask    = (rand < mask_ratio).float()
    mask_2d = mask.reshape(B, 1, ph, pw)
    mask_2d = mask_2d.repeat_interleave(patch_size, dim=2).repeat_interleave(patch_size, dim=3)
    return x * (1 - mask_2d), mask_2d

# ── Decoder — intentionally weak to force encoder to carry detail ──────────────

class MAEDecoder(nn.Module):
    """
    Minimal decoder: one bottleneck conv + bilinear upsample.
    No spatial hierarchy — encoder must encode spatial structure itself.
    """
    def __init__(self, in_channels, out_size):
        super().__init__()
        self.proj    = nn.Conv2d(in_channels, 512, kernel_size=3, padding=1, bias=False)
        self.proj1    = nn.Conv2d(512, 256, kernel_size=3, padding=1, bias=False)
        self.proj2    = nn.Conv2d(256, 128, kernel_size=3, padding=1, bias=False)
        self.out     = nn.Conv2d(128, 1, kernel_size=3, padding=1, bias=False)
        self.out_size = out_size
        
        self.up = nn.Upsample(scale_factor=2)

    def forward(self, x):
        x = self.up(F.gelu(self.proj(x)))
        x = self.up(F.gelu(self.proj1(x)))
        x = self.up(F.gelu(self.proj2(x)))
        x = self.up(F.gelu(self.proj3(x)))
        
        x = F.interpolate(x, size=(self.out_size, self.out_size),
                          mode="bilinear", align_corners=False)
        return self.out(x)

# ── LR schedule ────────────────────────────────────────────────────────────────

def get_lr(step, base_lr, warmup, total):
    if step < warmup:
        return base_lr * step / max(1, warmup)
    p = (step - warmup) / max(1, total - warmup)
    return base_lr * 0.5 * (1 + math.cos(math.pi * p))

# ── Reconstruction grid ────────────────────────────────────────────────────────

def denorm(t):
    return (t * 0.226 + 0.449).clamp(0, 1)

@torch.no_grad()
def save_recon_grid(model, dataset, patch_size, mask_ratio, epoch, out_dir, device, n=5):
    model.eval()
    imgs      = torch.stack([dataset[i][0] for i in random.sample(range(len(dataset)), n)]).to(device)
    masked, _ = mask_patches(imgs, patch_size, mask_ratio)
    recon     = model["decoder"](model["encoder"](masked))

    fig, axes = plt.subplots(3, n, figsize=(n * 3, 9), facecolor="#0f1117")
    fig.suptitle(f"MAE Reconstruction — Epoch {epoch}", color="#e2e8f0",
                 fontsize=13, fontweight="bold")

    for row, (data, label) in enumerate(zip(
        [denorm(imgs).cpu().squeeze(1).numpy(),
         denorm(masked).cpu().squeeze(1).numpy(),
         denorm(recon).cpu().squeeze(1).numpy()],
        ["Original", "Masked Input", "Reconstruction"]
    )):
        for col in range(n):
            ax = axes[row, col]
            ax.imshow(data[col], cmap="gray", vmin=0, vmax=1)
            ax.set_xticks([]); ax.set_yticks([])
            for spine in ax.spines.values(): spine.set_edgecolor("#334155")
            if col == 0: ax.set_ylabel(label, color="#94a3b8", fontsize=9)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    path = out_dir / f"recon_epoch_{epoch:04d}.png"
    plt.savefig(path, dpi=120, bbox_inches="tight", facecolor="#0f1117")
    plt.close()
    return path

# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone",    default="convnext_large.fb_in22k_ft_in1k_384")
    parser.add_argument("--size",        type=int,   default=384)
    parser.add_argument("--patch-size",  type=int,   default=32)
    parser.add_argument("--mask-ratio",  type=float, default=0.75)
    parser.add_argument("--epochs",      type=int,   default=2)
    parser.add_argument("--batch-size",  type=int,   default=16)
    parser.add_argument("--lr",          type=float, default=1e-4)
    parser.add_argument("--consist-lam", type=float, default=0.05,
                        help="Weight for augmentation consistency loss")
    parser.add_argument("--msk-only",    action="store_true")
    parser.add_argument("--resume",      default=None)
    parser.add_argument("--workers",     type=int,   default=2)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    viz_dir = OUT_DIR / "recon_grids"
    viz_dir.mkdir(exist_ok=True)

    log_file = open(OUT_DIR / "pretrain.log", "a")
    def log(msg):
        tqdm.write(msg)
        print(msg, file=log_file, flush=True)

    log(f"\n{'='*60}")
    log(f"  backbone:    {args.backbone}")
    log(f"  size:        {args.size}  patch: {args.patch_size}  mask: {args.mask_ratio}")
    log(f"  epochs:      {args.epochs}  batch: {args.batch_size}  lr: {args.lr}")
    log(f"  consist_lam: {args.consist_lam}  msk-only: {args.msk_only}")
    log(f"  MURA:        normal studies only")
    log(f"{'='*60}")

    # ── data ──────────────────────────────────────────────────────────────────
    if args.msk_only:
        roots = DATA_DIRS["mura_normal"] + DATA_DIRS["msk"]
    else:
        roots = DATA_DIRS["mura_normal"] + DATA_DIRS["msk"] + DATA_DIRS["cxr"]

    dataset = XRayDataset(roots, args.size)
    loader  = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                         num_workers=args.workers, pin_memory=True, drop_last=True)
    log(f"  {len(dataset):,} images  →  {len(loader):,} batches/epoch")

    # ── encoder ───────────────────────────────────────────────────────────────
    encoder = timm.create_model(args.backbone, pretrained=True, num_classes=0, global_pool="")

    stem_conv = None
    for name, mod in encoder.named_modules():
        if isinstance(mod, nn.Conv2d) and mod.in_channels == 3:
            stem_conv = (name, mod); break
    if stem_conv:
        name, old = stem_conv
        new = nn.Conv2d(1, old.out_channels, old.kernel_size, old.stride, old.padding,
                        bias=old.bias is not None)
        new.weight.data = old.weight.data.mean(dim=1, keepdim=True)
        if old.bias is not None: new.bias.data = old.bias.data.clone()
        parent = encoder
        for p in name.split(".")[:-1]: parent = getattr(parent, p)
        setattr(parent, name.split(".")[-1], new)

    with torch.no_grad():
        out = encoder(torch.zeros(1, 1, args.size, args.size))
    enc_channels = out.shape[1]
    log(f"  encoder output: {tuple(out.shape)}  ({enc_channels} ch)")

    decoder   = MAEDecoder(enc_channels, args.size)
    n_dec     = sum(p.numel() for p in decoder.parameters())
    n_enc     = sum(p.numel() for p in encoder.parameters())
    log(f"  encoder params: {n_enc:,}  decoder params: {n_dec:,}  ratio: {n_dec/n_enc:.4f}")

    model     = nn.ModuleDict({"encoder": encoder, "decoder": decoder}).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    scaler    = GradScaler()

    # ── resume ────────────────────────────────────────────────────────────────
    start_epoch, step = 1, 0
    if args.resume and Path(args.resume).exists():
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scaler.load_state_dict(ckpt["scaler"])
        start_epoch = ckpt["epoch"] + 1
        step        = ckpt["step"]
        log(f"  resumed from epoch {ckpt['epoch']}  (step {step}, loss {ckpt['loss']:.4f})")

    total_steps  = args.epochs * len(loader)
    warmup_steps = min(1000, total_steps // 20)
    log(f"  total steps: {total_steps:,}  warmup: {warmup_steps:,}\n")

    log_every = max(1, len(loader) // 10)

    # ── train ─────────────────────────────────────────────────────────────────
    for epoch in tqdm(range(start_epoch, args.epochs + 1), desc="Pretraining"):
        model.train()
        epoch_loss = epoch_recon = epoch_consist = 0.0
        window_loss = window_recon = window_consist = 0.0
        n_batches  = 0

        pbar = tqdm(loader, desc=f"Epoch {epoch:03d}", leave=False)
        for x, x_aug in pbar:
            x     = x.to(device)
            x_aug = x_aug.to(device)

            x_masked, mask = mask_patches(x, args.patch_size, args.mask_ratio)

            with autocast(device_type="cuda"):
                feats1 = model["encoder"](x_masked)
                feats2 = model["encoder"](x_aug)

                recon       = model["decoder"](feats1)
                loss_recon  = (F.mse_loss(recon, x, reduction="none") * mask).sum() \
                              / (mask.sum() + 1e-6)

                f1 = feats1.mean(dim=(2, 3))
                f2 = feats2.mean(dim=(2, 3))
                loss_consist = 1 - F.cosine_similarity(f1, f2).mean()

                loss = loss_recon + args.consist_lam * loss_consist

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

            step += 1
            lr = get_lr(step, args.lr, warmup_steps, total_steps)
            for pg in optimizer.param_groups: pg["lr"] = lr

            epoch_loss     += loss.item()
            epoch_recon    += loss_recon.item()
            epoch_consist  += loss_consist.item()
            window_loss    += loss.item()
            window_recon   += loss_recon.item()
            window_consist += loss_consist.item()
            n_batches      += 1

            pbar.set_postfix(recon=f"{loss_recon.item():.4f}",
                             consist=f"{loss_consist.item():.4f}",
                             lr=f"{lr:.2e}")

            if n_batches % log_every == 0:
                frac = n_batches / len(loader)
                log(f"  [{epoch:03d} {frac:5.1%}] "
                    f"recon {window_recon/log_every:.4f}  "
                    f"consist {window_consist/log_every:.4f}  "
                    f"total {window_loss/log_every:.4f}  lr {lr:.2e}")
                window_loss = window_recon = window_consist = 0.0
                torch.save({
                    "epoch": epoch, "step": step, "loss": epoch_loss / n_batches,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scaler": scaler.state_dict(),
                    "backbone": args.backbone,
                }, OUT_DIR / "mae_checkpoint.pt")

                grid_path = save_recon_grid(model, dataset, args.patch_size,
                                            args.mask_ratio, epoch, n_batches, viz_dir, device)
                log(f"  ↳ recon grid -> {grid_path}")
                model.train()

        avg_loss    = epoch_loss    / n_batches
        avg_recon   = epoch_recon   / n_batches
        avg_consist = epoch_consist / n_batches
        log(f"Epoch {epoch:03d} DONE | loss {avg_loss:.4f} "
            f"| recon {avg_recon:.4f}  consist {avg_consist:.4f} | lr {lr:.2e}")

    encoder_path = OUT_DIR / f"mae_encoder_{Path(args.backbone).stem}.pt"
    torch.save(model["encoder"].state_dict(), encoder_path)
    log(f"\nDone. Encoder weights → {encoder_path}")
    log_file.close()


if __name__ == "__main__":
    main()