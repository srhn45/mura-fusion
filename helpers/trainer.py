import warnings
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import cohen_kappa_score
from tqdm.auto import tqdm
from torch.amp import autocast, GradScaler

from helpers.unfreezer import ProgressiveUnfreezer

def focal_bce_loss(logits, labels, pos_weight=None, gamma=1.5):
    bce = F.binary_cross_entropy_with_logits(
        logits, labels,
        pos_weight=pos_weight,
        reduction="none"
    )
    p_t = torch.where(labels == 1, logits.sigmoid(), 1 - logits.sigmoid())
    focal_weight = (1 - p_t).clamp(min=1e-6) ** gamma
    return (focal_weight * bce).mean()

def get_lr(step, base_lr, warmup_steps, total_steps):
    if step < warmup_steps:
        return base_lr * step / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return base_lr * 0.5 * (1 + math.cos(math.pi * progress))

def _state_path(checkpoint_path):
    return checkpoint_path.replace(".pt", "_state.pt")

def save_training_state(checkpoint_path, epoch, global_step, bad_epochs,
                        best_kappa, optimizer, scaler, unfreezer, reporter):
    state = {
        "epoch":        epoch,
        "global_step":  global_step,
        "bad_epochs":   bad_epochs,
        "best_kappa":   best_kappa,
        "optimizer":    optimizer.state_dict(),
        "scaler":       scaler.state_dict(),
    }
    if unfreezer is not None:
        state["unfreezer"] = {
            "_wait":      unfreezer._wait,
            "_best_val":  unfreezer._best_val,
            "next_idx":   unfreezer.next_idx,
        }
    if reporter is not None:
        state["reporter"] = {
            "epochs":       reporter.epochs,
            "train_loss":   reporter.train_loss,
            "train_acc":    reporter.train_acc,
            "val_loss":     reporter.val_loss,
            "val_acc":      reporter.val_acc,
            "kappa":        reporter.kappa,
            "tuned_kappa":  reporter.tuned_kappa,
            "tuned_thresh": reporter.tuned_thresh,
            "cat_kappa":    reporter.cat_kappa,
            "cat_acc":      reporter.cat_acc,
            "best_epoch":   reporter.best_epoch,
            "best_kappa":   reporter.best_kappa,
        }
    torch.save(state, _state_path(checkpoint_path))

def load_training_state(checkpoint_path, optimizer, scaler, unfreezer, reporter, device):
    path = _state_path(checkpoint_path)
    state = torch.load(path, map_location=device, weights_only=False)

    optimizer.load_state_dict(state["optimizer"])
    scaler.load_state_dict(state["scaler"])

    if unfreezer is not None and "unfreezer" in state:
        u = state["unfreezer"]
        unfreezer._wait     = u["_wait"]
        unfreezer._best_val = u["_best_val"]
        unfreezer.next_idx  = u["next_idx"]

    if reporter is not None and "reporter" in state:
        r = state["reporter"]
        reporter.epochs       = r["epochs"]
        reporter.train_loss   = r["train_loss"]
        reporter.train_acc    = r["train_acc"]
        reporter.val_loss     = r["val_loss"]
        reporter.val_acc      = r["val_acc"]
        reporter.kappa        = r["kappa"]
        reporter.tuned_kappa  = r["tuned_kappa"]
        reporter.tuned_thresh = r["tuned_thresh"]
        reporter.cat_kappa    = r["cat_kappa"]
        reporter.cat_acc      = r["cat_acc"]
        reporter.best_epoch   = r["best_epoch"]
        reporter.best_kappa   = r["best_kappa"]

    return state["epoch"], state["global_step"], state["bad_epochs"], state["best_kappa"]


@torch.no_grad()
def evaluate(model, loader, device, pos_weight=None, log=tqdm.write):
    model.eval()
    criterion = nn.BCEWithLogitsLoss()

    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels, all_probs = [], [], []
    cat_preds, cat_labels = {}, {}

    for image_list, labels, categories in tqdm(loader, desc="Validating", leave=False):
        labels = labels.to(device)
        with autocast(device_type="cuda"):
            logits, _ = model(image_list, categories)
            loss = criterion(logits, labels)

        total_loss += loss.item() * labels.size(0)
        preds = (logits.sigmoid() > 0.5).float()
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        all_preds.extend(preds.cpu().int().tolist())
        all_labels.extend(labels.cpu().int().tolist())
        all_probs.extend(logits.sigmoid().cpu().float().tolist())

        for p, l, c in zip(preds.cpu().int().tolist(), labels.cpu().int().tolist(), categories):
            cat_preds.setdefault(c, []).append(p)
            cat_labels.setdefault(c, []).append(l)

    thresholds  = np.arange(0.3, 0.7, 0.02)
    probs_arr   = np.array(all_probs)
    labels_arr  = np.array(all_labels)
    best_t, best_k = 0.5, -1.0
    for t in thresholds:
        k = cohen_kappa_score(labels_arr, (probs_arr > t).astype(int))
        if k > best_k:
            best_k, best_t = k, t

    kappa = cohen_kappa_score(all_labels, all_preds)
    per_cat = {c: (cohen_kappa_score(cat_labels[c], cat_preds[c]),
                   sum(p == l for p, l in zip(cat_preds[c], cat_labels[c])) / len(cat_labels[c]))
               for c in cat_labels}

    ordered_cats = ["XR_SHOULDER", "XR_HUMERUS", "XR_ELBOW",
                    "XR_FOREARM",  "XR_WRIST",   "XR_HAND", "XR_FINGER"]

    log("\n  Per-category results:")
    for c in ordered_cats:
        if c in per_cat:
            k_c, a_c = per_cat[c]
            log(f"   {c.replace('XR_',''):>8} | κ={k_c:5.2f} | acc={a_c:5.2f}")
        else:
            log(f"   {c.replace('XR_',''):>8} | κ=  N/A | acc=  N/A")

    log(f"\n  Tuned results:")
    log(f"  threshold={best_t:.2f}  tuned_kappa={best_k:.4f}  untuned_kappa={kappa:.4f}")

    return total_loss / total, correct / total, kappa, best_k, best_t, per_cat


def fit(
    model,
    train_loader,
    val_loader,
    n_epochs=30,
    lr=1e-4,
    lr_decay_threshold=1e-3,
    weight_decay=1e-2,
    pos_weight=None,
    device="cuda",
    warmup_steps=500,
    unfreeze_groups=None,
    unfreeze_patience=2,
    unfreeze_lr_scale=0.1,
    checkpoint_path="model/best_model.pt",
    save_fn=None,
    reporter=None,
    resume=False,       # set True to restore full training state
):
    model = model.to(device)

    log_path = checkpoint_path.replace(".pt", ".log")
    log_file = open(log_path, "a")

    def log(msg):
        tqdm.write(msg)
        print(msg, file=log_file, flush=True)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr, weight_decay=weight_decay
    )

    try:
        log("\nBackbone freeze status:")
        for i, layer in enumerate(model.backbone.backbone.encoder.layers):
            status = "trainable" if any(p.requires_grad for p in layer.parameters()) else "frozen"
            log(f"encoder_layer_{i:02d} -> {status}")
        ln_status = "trainable" if any(p.requires_grad for p in model.backbone.backbone.encoder.ln.parameters()) else "frozen"
        log(f"encoder_ln -> {ln_status}")
        conv_status = "trainable" if any(p.requires_grad for p in model.backbone.backbone.conv_proj.parameters()) else "frozen"
        log(f"conv_proj -> {conv_status}")
    except AttributeError:
        total_trainable = sum(p.numel() for p in model.backbone.parameters() if p.requires_grad)
        total_frozen    = sum(p.numel() for p in model.backbone.parameters() if not p.requires_grad)
        log(f"  trainable: {total_trainable:,}  frozen: {total_frozen:,}")

    total_steps = n_epochs * len(train_loader)
    scaler      = GradScaler()

    unfreezer = None
    if unfreeze_groups:
        unfreezer = ProgressiveUnfreezer(
            optimizer, unfreeze_groups, lr_scale=unfreeze_lr_scale,
            patience=unfreeze_patience, prescheduled=True, mode="max"
        )

    # defaults — overwritten if resuming
    start_epoch  = 1
    global_step  = 0
    bad_epochs   = 0
    best_kappa   = 0.0

    if resume:
        try:
            start_epoch, global_step, bad_epochs, best_kappa = load_training_state(
                checkpoint_path, optimizer, scaler, unfreezer, reporter, device
            )
            start_epoch += 1   # resume from next epoch
            log(f"  ↳ Resumed from epoch {start_epoch - 1}  (global_step={global_step})")
        except FileNotFoundError:
            log("  ↳ No training state found, starting from scratch")

    for epoch in tqdm(range(start_epoch, n_epochs + 1), desc="Training"):
        # ── train ──────────────────────────────────────────────────────────────
        model.train()
        train_loss, correct, total = 0.0, 0, 0

        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch:03d} Train", leave=False)
        for image_list, labels, categories in train_pbar:
            labels = labels.to(device)
            pw = torch.tensor([pos_weight[cat] for cat in categories], device=device) \
                 if isinstance(pos_weight, dict) else None

            with autocast(device_type="cuda"):
                logits, _ = model(image_list, categories)
                loss = focal_bce_loss(logits, labels, pos_weight=pw, gamma=1.5)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

            global_step += 1
            current_lr = get_lr(global_step, lr, warmup_steps, total_steps)
            for pg in optimizer.param_groups:
                pg['lr'] = current_lr * pg.get('lr_scale', 1.0)

            train_loss += loss.item() * labels.size(0)
            preds = (logits.detach().sigmoid() > 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            train_pbar.set_postfix({'loss': f"{train_loss/total:.4f}",
                                    'acc':  f"{correct/total:.4f}"})

        train_loss /= total
        train_acc = correct / total

        # ── validate ───────────────────────────────────────────────────────────
        val_loss, val_acc, kappa, tuned_kappa, tuned_thresh, per_cat = evaluate(
            model, val_loader, device, pos_weight=pos_weight, log=log
        )

        log(f"Epoch {epoch:03d} | "
            f"train loss {train_loss:.4f}  acc {train_acc:.4f} | "
            f"val loss {val_loss:.4f}  acc {val_acc:.4f}  | "
            f"kappa {kappa:.4f}")

        if reporter:
            reporter.log_epoch(epoch, train_loss, train_acc, val_loss, val_acc,
                               kappa, tuned_kappa, tuned_thresh, per_cat)

        # ── checkpointing ──────────────────────────────────────────────────────
        if kappa > best_kappa + lr_decay_threshold:
            best_kappa = kappa
            bad_epochs = 0
            log(f"  ↳ checkpoint saved (val_loss={val_loss:.4f}, kappa={kappa:.4f})")
            if save_fn:
                save_fn(model)
                save_training_state(checkpoint_path, epoch, global_step, bad_epochs,
                    best_kappa, optimizer, scaler, unfreezer, reporter)
            else:
                torch.save(model.state_dict(), checkpoint_path)
        else:
            bad_epochs += 1
            if bad_epochs >= int(20 * unfreeze_patience):
                log(f"  ↳ Reversal patience reached, reverting to best model")
                ckpt = torch.load(checkpoint_path)
                model.load_state_dict(ckpt["state_dict"] if "state_dict" in ckpt else ckpt)
                if unfreezer:
                    unfreezer._wait      = 0
                    unfreezer._best_val  = best_kappa
                bad_epochs = 0

        if unfreezer and not unfreezer.all_unfrozen():
            if unfreezer.step(kappa):
                bad_epochs = 0


    log(f"\nTraining complete. Best kappa: {best_kappa:.4f}")
    log_file.close()

    if reporter:
        reporter.save()

    return model