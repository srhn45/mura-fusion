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

def focal_bce_loss(logits, labels, pos_weight=None, gamma=1.2):
    bce = F.binary_cross_entropy_with_logits(
        logits, labels,
        pos_weight=pos_weight,
        reduction="none"
    ) # standard BCE
    # focal weight: (1 - p_t)^gamma
    p_t = torch.where(labels == 1, logits.sigmoid(), 1 - logits.sigmoid())
    focal_weight = (1 - p_t).clamp(min=1e-6) ** gamma
    return (focal_weight * bce).mean()

def get_lr(step, base_lr, warmup_steps, total_steps):
    if step < warmup_steps:
        return base_lr * step / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return base_lr * 0.5 * (1 + math.cos(math.pi * progress))

@torch.no_grad()
def evaluate(model, loader, device, pos_weight=None, log=tqdm.write):
    model.eval()
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weight], device=device) if pos_weight else None
    )
    
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []
    cat_preds, cat_labels = {}, {}
    
    all_probs = [] #!

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
        
        all_probs.extend(logits.sigmoid().cpu().float().tolist()) #!

        for p, l, c in zip(preds.cpu().int().tolist(), labels.cpu().int().tolist(), categories):
            cat_preds.setdefault(c, []).append(p)
            cat_labels.setdefault(c, []).append(l)
        
    thresholds = np.arange(0.3, 0.7, 0.02)
    best_t, best_k = 0.5, -1
    for t in thresholds:
        preds_t = [int(p > t) for p in all_probs]
        try:
            k = cohen_kappa_score(all_labels, preds_t)
            if k > best_k:
                best_k, best_t = k, t
        except:
            pass

    kappa = cohen_kappa_score(all_labels, all_preds)
    per_cat = {c: (cohen_kappa_score(cat_labels[c], cat_preds[c]),
                   sum(p == l for p, l in zip(cat_preds[c], cat_labels[c])) / len(cat_labels[c]))
               for c in cat_labels}

    # fixed category order
    ordered_cats = [
        "XR_SHOULDER", "XR_HUMERUS", "XR_ELBOW",
        "XR_FOREARM", "XR_WRIST", "XR_HAND", "XR_FINGER"
    ]

    rows = []
    for c in ordered_cats:
        if c in per_cat:
            kappa_c, acc_c = per_cat[c]
            rows.append(f"{c.replace('XR_', ''):>8} | κ={kappa_c:5.2f} | acc={acc_c:5.2f}")
        else:
            rows.append(f"{c.replace('XR_', ''):>8} | κ=  N/A | acc=  N/A")

    log("\n  Per-category results:")
    for r in rows:
        log("   " + r)

    log("\n  Tuned results:")
    log(f"  threshold={best_t:.2f}  tuned_kappa={best_k:.4f}  untuned_kappa={kappa:.4f}")

    return total_loss / total, correct / total, kappa

def fit(
    model,
    train_loader,
    val_loader,
    n_epochs=30,
    lr=1e-4,
    lr_decay_threshold=1e-3,
    weight_decay=1e-2,
    pos_weight=1.47,
    device="cuda",
    warmup_steps=500, 
    unfreeze_groups=None,
    unfreeze_patience=2,
    unfreeze_lr_scale=0.1,
    checkpoint_path="model/best_model.pt",
    save_fn=None
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
        log("\nBackbone freeze status:")
        total_trainable = sum(p.numel() for p in model.backbone.parameters() if p.requires_grad)
        total_frozen    = sum(p.numel() for p in model.backbone.parameters() if not p.requires_grad)
        log(f"  trainable: {total_trainable:,}  frozen: {total_frozen:,}")
    
    total_steps = n_epochs * len(train_loader)
    #def lr_lambda(step):
    #    if step < warmup_steps:
    #        return step / max(1, warmup_steps) # linear warmup
    #    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    #    return 0.5 * (1 + math.cos(math.pi * progress)) # cosine decay

    #with warnings.catch_warnings():
    #    warnings.simplefilter("ignore", UserWarning)
    #    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    global_step = 0
    
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weight], device=device) if pos_weight else None
    )
    
    scaler = GradScaler()

    unfreezer = None
    if unfreeze_groups:
        unfreezer = ProgressiveUnfreezer(
            optimizer, unfreeze_groups, lr_scale=unfreeze_lr_scale, 
            patience=unfreeze_patience, 
            prescheduled=True,
            mode="max"
        )
        
    bad_epochs = 0
    best_kappa = 0

    for epoch in tqdm(range(1, n_epochs + 1), desc="Training"):
        # ── train ──────────────────────────────────────────────────────
        model.train()
        train_loss, correct, total = 0.0, 0, 0
        
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch:03d} Train", leave=False)
        for image_list, labels, categories in train_pbar:
            labels = labels.to(device)
            
            with autocast(device_type="cuda"):
                logits, _ = model(image_list, categories)
                loss = focal_bce_loss(logits, labels, pos_weight=criterion.pos_weight, gamma=1.2)
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            
            global_step += 1
            current_lr = get_lr(global_step, lr, warmup_steps, total_steps)
            for pg in optimizer.param_groups:
                pg['lr'] = current_lr * pg.get('lr_scale', 1.0)
            
            train_loss += loss.item() * labels.size(0)
            preds = (logits.detach().sigmoid() > 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
            train_pbar.set_postfix({
                'loss': f"{train_loss/total:.4f}",
                'acc': f"{correct/total:.4f}"
            })
            
        train_loss /= total
        train_acc = correct / total

        # ── validate ───────────────────────────────────────────────────
        val_loss, val_acc, kappa = evaluate(model, val_loader, device, pos_weight=pos_weight,
                                            log=log)

        log(
            f"Epoch {epoch:03d} | "
            f"train loss {train_loss:.4f}  acc {train_acc:.4f} | "
            f"val loss {val_loss:.4f}  acc {val_acc:.4f}  | "
            f"kappa {kappa:.4f}"
        )

        # ── schedulers & checkpointing ─────────────────────────────────        
        if kappa > best_kappa + lr_decay_threshold:
            best_kappa = kappa
            bad_epochs = 0
            log(f"  ↳ checkpoint saved (val_loss={val_loss:.4f}, kappa={kappa:.4f})")
            if save_fn:
                save_fn(model)
            else:
                torch.save(model.state_dict(), checkpoint_path)
        else:
            bad_epochs += 1
            
            if bad_epochs >= int(20 * unfreeze_patience):
                log(f"  ↳ Reversal patience reached, reverting to best model")
                ckpt = torch.load(checkpoint_path)
                model.load_state_dict(ckpt["state_dict"] if "state_dict" in ckpt else ckpt)
                if unfreezer:
                    unfreezer._wait = 0
                    unfreezer._best_val = best_kappa
                bad_epochs = 0
        
        if unfreezer and not unfreezer.all_unfrozen():
            stepped = unfreezer.step(kappa)
            
            if stepped:
                bad_epochs = 0

    log(f"\nTraining complete. Best kappa: {best_kappa:.4f}")
    log_file.close()
    return model