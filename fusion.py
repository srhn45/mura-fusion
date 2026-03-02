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
import pandas as pd

from architectures.vit_l_16 import ViT_L_16_Backbone
from architectures.classifier import Classifier
from helpers.patientdataset import load_df, PatientDataset, patient_collate_fn, make_loader
from helpers.checkpoint import save_checkpoint
from helpers.augments import make_transform
from helpers.trainer import fit

import warnings # Custom RMSNorm is float32
warnings.filterwarnings("ignore", message="Mismatch dtype between input and weight")

# ── Config ────────────────────────────────────────────────────────────────────

CATEGORIES = ["XR_SHOULDER", "XR_HUMERUS", "XR_ELBOW",
              "XR_FOREARM", "XR_WRIST", "XR_HAND", "XR_FINGER"]

DATA_DIR       = "data/MURA-v1.1"
PARENT_DIR     = "data"
CHECKPOINT     = "models/best_model_vit_l_16.pt"
BACKBONE_KWARGS    = dict(embed_dim=256, freeze_until="encoder_layer_22", dropout=0.1)
CLASSIFIER_KWARGS = dict(embed_dim=256, mlp_depth=2, categories=CATEGORIES)
FIT_KWARGS = dict(
    n_epochs=50, lr=1e-4, pos_weight=1.47,
    unfreeze_patience=3, unfreeze_lr_scale=0.5,
)

# ── Data ──────────────────────────────────────────────────────────────────────

train_loader = make_loader(load_df("train_image_paths.csv", DATA_DIR), augment=True,
                           parent_dir=PARENT_DIR, size=512,
                           batch_size=8, shuffle=True, num_workers=2, pin_memory=True,
                           drop_last=True, persistent_workers=False)

val_loader   = make_loader(load_df("valid_image_paths.csv", DATA_DIR), augment=False,
                           parent_dir=PARENT_DIR, size=512,
                           batch_size=8, shuffle=False, num_workers=2, pin_memory=True,
                           persistent_workers=False)

# ── Model ─────────────────────────────────────────────────────────────────────

backbone = ViT_L_16_Backbone(**BACKBONE_KWARGS)
model    = Classifier(backbone, **CLASSIFIER_KWARGS)

unfreeze_groups = [*backbone.backbone.encoder.layers[-9:], 
                   backbone.backbone.encoder.ln,
                   backbone.backbone.class_token]

def save_fn(model):
    save_checkpoint(model, CHECKPOINT, backbone_cls=type(backbone).__name__,
                    backbone_kwargs=BACKBONE_KWARGS, classifier_kwargs=CLASSIFIER_KWARGS,
                    **FIT_KWARGS)
    
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total parameters: {total_params:,}")
print(f"Initially trainable parameters: {trainable_params:,}")

# ── Train ─────────────────────────────────────────────────────────────────────

model = fit(model, train_loader, val_loader,
            unfreeze_groups=unfreeze_groups,
            checkpoint_path=CHECKPOINT,
            save_fn=save_fn,
            **FIT_KWARGS)