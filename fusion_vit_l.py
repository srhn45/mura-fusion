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
from architectures.vit_l_16 import ViT_L_16_Backbone
from architectures.classifier import Classifier
from helpers.patientdataset import load_df, make_loader
from helpers.checkpoint import save_checkpoint, load_checkpoint
from helpers.trainer import fit
from helpers.reporter import Reporter

warnings.filterwarnings("ignore", message="Mismatch dtype between input and weight")

# ── Config ────────────────────────────────────────────────────────────────────

CATEGORIES = ["XR_SHOULDER", "XR_HUMERUS", "XR_ELBOW",
              "XR_FOREARM", "XR_WRIST", "XR_HAND", "XR_FINGER"]

DATA_DIR          = "data/MURA-v1.1"
PARENT_DIR        = "data"
CHECKPOINT        = "models/vit_l_16/best_model_vit_l_16.pt"
BACKBONE_KWARGS   = dict(embed_dim=256, freeze_until="encoder_layer_0", dropout=0.1, finetune_input=True)
CLASSIFIER_KWARGS = dict(embed_dim=256, mlp_depth=2, categories=CATEGORIES)
FIT_KWARGS        = dict(n_epochs=50, lr=1e-5, pos_weight=1.47, unfreeze_patience=3, unfreeze_lr_scale=0.1)

# ── Resume (comment out to train from scratch) ────────────────────────────────
#RESUME_FROM     = "models/best_model_vit_l_16.pt"
#RESUME_FREEZE   = "encoder_layer_19"   # override freeze depth on load (optional)

# ── Data ──────────────────────────────────────────────────────────────────────

train_loader = make_loader(load_df("train_image_paths.csv", DATA_DIR), augment=True,
                           parent_dir=PARENT_DIR, size=224, batch_size=8,
                           shuffle=True, num_workers=2, pin_memory=True,
                           drop_last=True, persistent_workers=False)
val_loader   = make_loader(load_df("valid_image_paths.csv", DATA_DIR), augment=False,
                           parent_dir=PARENT_DIR, size=224, batch_size=8,
                           shuffle=False, num_workers=2, pin_memory=True,
                           persistent_workers=False)

# ── Model ─────────────────────────────────────────────────────────────────────

try:
    model, ckpt_config = load_checkpoint(RESUME_FROM, device="cuda")
    # optionally override freeze depth after loading
    if "RESUME_FREEZE" in dir():
        for i, layer in enumerate(model.backbone.backbone.encoder.layers):
            freeze_idx = ViT_L_16_Backbone.__init__.__defaults__  # resolved below
        freeze_idx = {"encoder_layer_0": 0, "encoder_layer_2": 2, "encoder_layer_4": 4,
                      "encoder_layer_6": 6, "encoder_layer_8": 8, "encoder_layer_10": 10,
                      "encoder_layer_12": 12, "encoder_layer_14": 14, "encoder_layer_16": 16,
                      "encoder_layer_18": 18, "encoder_layer_20": 20, "encoder_layer_22": 22,
                      "encoder_layer_24": 24}.get(RESUME_FREEZE, 24)
        for i, layer in enumerate(model.backbone.backbone.encoder.layers):
            for p in layer.parameters():
                p.requires_grad = i >= freeze_idx
    backbone = model.backbone
    print(f"Resumed from {RESUME_FROM}")
except NameError:  # RESUME_FROM undefined - fresh start
    backbone = ViT_L_16_Backbone(**BACKBONE_KWARGS)
    model    = Classifier(backbone, **CLASSIFIER_KWARGS)

unfreeze_groups = [
    *reversed(backbone.backbone.encoder.layers[-24:]),
    backbone.backbone.encoder.ln,
    backbone.backbone.class_token,
    backbone.backbone.conv_proj,
]

def save_fn(model):
    save_checkpoint(model, CHECKPOINT, backbone_cls=type(backbone).__name__,
                    backbone_kwargs=BACKBONE_KWARGS, classifier_kwargs=CLASSIFIER_KWARGS,
                    **FIT_KWARGS)

total_params     = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total parameters:           {total_params:,}")
print(f"Initially trainable params: {trainable_params:,}")

# ── Train ─────────────────────────────────────────────────────────────────────

reporter = Reporter(
    checkpoint_path  = CHECKPOINT,
    model_name       = "ViT-L/16",
    backbone_kwargs  = BACKBONE_KWARGS,
    classifier_kwargs= CLASSIFIER_KWARGS,
    fit_kwargs       = FIT_KWARGS,
)

model = fit(model, train_loader, val_loader,
            unfreeze_groups=unfreeze_groups,
            checkpoint_path=CHECKPOINT,
            save_fn=save_fn,
            reporter=reporter,
            **FIT_KWARGS)