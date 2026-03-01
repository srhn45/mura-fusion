import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import cohen_kappa_score
from tqdm.auto import tqdm

from helpers.unfreezer import ProgressiveUnfreezer

@torch.no_grad()
def evaluate(model, loader, device, pos_weight=None):
    model.eval()
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weight], device=device) if pos_weight else None
    )

    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []

    for image_list, labels in tqdm(loader, desc="Validating", leave=False):
        labels = labels.to(device)
        logits, _ = model(image_list)
        loss = criterion(logits, labels)

        total_loss += loss.item() * labels.size(0)
        preds = (logits.sigmoid() > 0.5).float()
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        all_preds.extend(preds.cpu().int().tolist())
        all_labels.extend(labels.cpu().int().tolist())

    kappa = cohen_kappa_score(all_labels, all_preds)
    return total_loss / total, correct / total, kappa

def fit(
    model,
    train_loader,
    val_loader,
    n_epochs=30,
    lr=1e-4,
    lr_decay_threshold=1e-2,
    weight_decay=1e-4,
    pos_weight=1.47,
    device="cuda",
    scheduler_patience=3,
    unfreeze_groups=None,
    unfreeze_patience=2,
    unfreeze_lr_scale=0.5,
    checkpoint_path="model/best_model.pt",
    save_fn=None
):
    model = model.to(device)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr, weight_decay=weight_decay
    )
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=scheduler_patience, threshold=lr_decay_threshold
    )
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weight], device=device) if pos_weight else None
    )

    unfreezer = None
    if unfreeze_groups:
        unfreezer = ProgressiveUnfreezer(
            optimizer, unfreeze_groups, lr_scale=0.5, 
            patience=unfreeze_patience, mode="max"
        )
        
    bad_epochs = 0
    best_kappa = 0

    for epoch in tqdm(range(1, n_epochs + 1), desc="Training"):
        # ── train ──────────────────────────────────────────────────────
        model.train()
        train_loss, correct, total = 0.0, 0, 0
        
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch:03d} Train", leave=False)
        for image_list, labels in train_pbar:
            labels = labels.to(device)
            optimizer.zero_grad()
            logits, _ = model(image_list)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
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
        val_loss, val_acc, kappa = evaluate(model, val_loader, device, pos_weight=pos_weight)

        tqdm.write(
            f"Epoch {epoch:03d} | "
            f"train loss {train_loss:.4f}  acc {train_acc:.4f} | "
            f"val loss {val_loss:.4f}  acc {val_acc:.4f}  | "
            f"kappa {kappa:.4f}"
        )

        # ── schedulers & checkpointing ─────────────────────────────────        
        if kappa > best_kappa + lr_decay_threshold:
            best_kappa = kappa
            bad_epochs = 0
            tqdm.write(f"  ↳ checkpoint saved (val_loss={val_loss:.4f}, kappa={kappa:.4f})")
            if save_fn:
                save_fn(model)
            else:
                torch.save(model.state_dict(), checkpoint_path)
        else:
            bad_epochs += 1
            
            if bad_epochs >= 2 * max(unfreeze_patience, scheduler_patience):
                tqdm.write(f"  ↳ Reversal patience reached, reverting to best model")
                ckpt = torch.load(checkpoint_path)
                model.load_state_dict(ckpt["state_dict"] if "state_dict" in ckpt else ckpt)
                if unfreezer:
                    unfreezer._wait = 0
                    unfreezer._best_val = best_kappa
                bad_epochs = 0
        
        lr_scheduler.step(kappa)
        if unfreezer and not unfreezer.all_unfrozen():
            stepped = unfreezer.step(kappa)
            
            if stepped:
                bad_epochs = 0

    print(f"\nTraining complete. Best kappa: {best_kappa:.4f}")
    return model