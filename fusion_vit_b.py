import signal, sys
def cleanup(sig, frame):
    print("\nCleaning up...")
    try:
        train_loader._iterator = None
        val_loader._iterator   = None
    except:
        pass
    torch.cuda.empty_cache()
    sys.exit(0)

signal.signal(signal.SIGINT,  cleanup)
signal.signal(signal.SIGTERM, cleanup)

# ──────────────────────────────────────────────────────────────────────────────
import os
import torch
import warnings
from architectures.vit_b_16 import ViT_B_16_Backbone
from architectures.classifier import Classifier
from helpers.patientdataset import load_df, make_loader
from helpers.checkpoint import save_checkpoint, load_checkpoint
from helpers.trainer import fit

warnings.filterwarnings("ignore", message="Mismatch dtype between input and weight")

# ── Config ────────────────────────────────────────────────────────────────────

CATEGORIES = ["XR_SHOULDER", "XR_HUMERUS", "XR_ELBOW",
              "XR_FOREARM", "XR_WRIST", "XR_HAND", "XR_FINGER"]

DATA_DIR          = "data/MURA-v1.1"
PARENT_DIR        = "data"
CHECKPOINT        = "models/best_model_vit_b_16.pt"

BACKBONE_KWARGS   = dict(
    embed_dim=256,
    freeze_until="encoder_layer_0",
    dropout=0.1,
    finetune_input=True
)

CLASSIFIER_KWARGS = dict(
    embed_dim=256,
    mlp_depth=2,
    categories=CATEGORIES
)

FIT_KWARGS        = dict(
    n_epochs=30,
    lr=1e-5,
    pos_weight=1.47,
    unfreeze_patience=3,
    unfreeze_lr_scale=0.1
)

# ── Data ──────────────────────────────────────────────────────────────────────

train_loader = make_loader(
    load_df("train_image_paths.csv", DATA_DIR),
    augment=True,
    parent_dir=PARENT_DIR,
    size=384,
    batch_size=16,
    shuffle=True,
    num_workers=2,
    pin_memory=True,
    drop_last=True,
    persistent_workers=False
)

val_loader = make_loader(
    load_df("valid_image_paths.csv", DATA_DIR),
    augment=False,
    parent_dir=PARENT_DIR,
    size=384,
    batch_size=16,
    shuffle=False,
    num_workers=2,
    pin_memory=True,
    persistent_workers=False
)

# ── Model ─────────────────────────────────────────────────────────────────────

try:
    model, ckpt_config = load_checkpoint(RESUME_FROM, device="cuda")

    if "RESUME_FREEZE" in dir():
        freeze_idx = {
            "encoder_layer_0": 0,
            "encoder_layer_2": 2,
            "encoder_layer_4": 4,
            "encoder_layer_6": 6,
            "encoder_layer_8": 8,
            "encoder_layer_10": 10,
            "encoder_layer_12": 12,
        }.get(RESUME_FREEZE, 12)

        for i, layer in enumerate(model.backbone.backbone.encoder.layers):
            for p in layer.parameters():
                p.requires_grad = i >= freeze_idx

    backbone = model.backbone
    print(f"Resumed from {RESUME_FROM}")

except NameError:
    backbone = ViT_B_16_Backbone(**BACKBONE_KWARGS)   # ← changed
    model    = Classifier(backbone, **CLASSIFIER_KWARGS)
    

unfreeze_groups = [
    *reversed(backbone.backbone.encoder.layers[-12:]),  # ← changed
    backbone.backbone.encoder.ln,
    backbone.backbone.class_token,
    backbone.backbone.conv_proj,
]

def save_fn(model):
    save_checkpoint(
        model,
        CHECKPOINT,
        backbone_cls=type(backbone).__name__,
        backbone_kwargs=BACKBONE_KWARGS,
        classifier_kwargs=CLASSIFIER_KWARGS,
        **FIT_KWARGS
    )

total_params     = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"Total parameters:           {total_params:,}")
print(f"Initially trainable params: {trainable_params:,}")

# ── Train ─────────────────────────────────────────────────────────────────────

model = fit(
    model,
    train_loader,
    val_loader,
    unfreeze_groups=unfreeze_groups,
    checkpoint_path=CHECKPOINT,
    save_fn=save_fn,
    **FIT_KWARGS
)