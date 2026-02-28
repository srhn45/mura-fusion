import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import cohen_kappa_score

from helpers.unfreezer import ProgressiveUnfreezer

@torch.no_grad()
def evaluate(model, loader, device, pos_weight=None):
    model.eval()
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weight], device=device) if pos_weight else None
    )

    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []

    for image_list, labels in loader:
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
    lr_decay_threshold=1e-3
    weight_decay=1e-4,
    pos_weight=1.47,
    device="cuda",
    scheduler_patience=3,
    unfreeze_groups=None,
    unfreeze_patience=2,
    unfreeze_lr_scale=0.5,
    checkpoint_path="model/best_model.pt"
):
    model = model.to(device)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr, weight_decay=weight_decay
    )
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=scheduler_patience, threshold=lr_decay_threshold
    )
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weight], device=device) if pos_weight else None
    )

    unfreezer = None
    if unfreeze_groups:
        unfreezer = ProgressiveUnfreezer(
            optimizer,
            unfreeze_groups,
            lr_scale=unfreeze_lr_scale,
            patience=unfreeze_patience,
        )
        
    bad_epochs = 0
    best_val_loss = float("inf")

    for epoch in range(1, n_epochs + 1):
        # ── train ──────────────────────────────────────────────────────
        model.train()
        train_loss, correct, total = 0.0, 0, 0
        for image_list, labels in train_loader:
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
        train_loss /= total
        train_acc = correct / total

        # ── validate ───────────────────────────────────────────────────
        model.eval()        
        val_loss, val_acc, kappa = evaluate(model, val_loader, device, pos_weight=pos_weight)

        print(
            f"Epoch {epoch:03d} | "
            f"train loss {train_loss:.4f}  acc {train_acc:.4f} | "
            f"val loss {val_loss:.4f}  acc {val_acc:.4f}  | "
            f"kappa {kappa}"
        )

        # ── schedulers ─────────────────────────────────────────────────        
        if val_loss > best_val_loss - lr_decay_threshold:
            bad_epochs += 1
            if bad_epochs >= min(scheduler_patience, unfreeze_patience):
                print(f"  ↳ Patience reached, reverting to best model before LR reduction/unfreezing")
                model.load_state_dict(torch.load(checkpoint_path))
                bad_epochs = 0
        else:
            bad_epochs = 0
        
        lr_scheduler.step(val_loss)
        if unfreezer and not unfreezer.all_unfrozen():
            unfreezer.step(val_loss)
                

        # ── checkpoint ─────────────────────────────────────────────────
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), checkpoint_path)
            print(f"  ↳ checkpoint saved (val_loss={val_loss:.4f})")

    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")
    return model