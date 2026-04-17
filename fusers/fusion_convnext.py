'''
PYTORCH_ALLOC_CONF=expandable_segments:True uv run fusion_convnext.py
'''

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

sys.path.insert(0, '..')
# ──────────────────────────────────────────────────────────────────────────────
import torch
import warnings
from architectures.convnext_l import ConvNeXt_L_Backbone
from architectures.classifier_separate import Classifier
from helpers.patientdataset import load_df, make_loader
from helpers.checkpoint import save_checkpoint, load_checkpoint
from helpers.trainer import fit
from helpers.reporter import Reporter

warnings.filterwarnings("ignore", message="Mismatch dtype between input and weight")
warnings.filterwarnings("ignore", message="Online softmax is disabled")
warnings.filterwarnings("ignore", message="Not enough SMs")

# ── Config ────────────────────────────────────────────────────────────────────

#PRETRAINED_ENCODER = "../models/pretrained/mae_encoder_convnext_large.pt"
PRETRAINED_ENCODER = None
# set to None to skip

CATEGORIES = ["XR_SHOULDER", "XR_HUMERUS", "XR_ELBOW",
              "XR_FOREARM", "XR_WRIST", "XR_HAND", "XR_FINGER"]

DATA_DIR          = "../data/MURA-v1.1"
PARENT_DIR        = "../data"
CHECKPOINT        = "../models/convnext_l/best_model_convnext_l.pt"
BACKBONE_KWARGS   = dict(embed_dim=256, freeze_until="stage0", dropout=0.1, finetune_input=True) 
                         #tiled=False)
CLASSIFIER_KWARGS = dict(embed_dim=256, mlp_depth=2, categories=CATEGORIES)
FIT_KWARGS        = dict(n_epochs=50, lr=1e-5, pos_weight=1.47, unfreeze_patience=3, unfreeze_lr_scale=0.1)

# ── Resume (comment out to train from scratch) ────────────────────────────────
#RESUME_FROM       = "../models/convnext_l/best_model_convnext_l.pt"

# ── Data ──────────────────────────────────────────────────────────────────────

train_loader = make_loader(load_df("train_image_paths.csv", DATA_DIR), augment=True,
                           parent_dir=PARENT_DIR, size=384, batch_size=8,
                           shuffle=True, num_workers=2, pin_memory=True,
                           drop_last=True, persistent_workers=False)
val_loader   = make_loader(load_df("valid_image_paths.csv", DATA_DIR), augment=False,
                           parent_dir=PARENT_DIR, size=384, batch_size=8,
                           shuffle=False, num_workers=2, pin_memory=True,
                           persistent_workers=False)

# ── Model ─────────────────────────────────────────────────────────────────────
import os

if "RESUME_FROM" in dir() and os.path.exists(RESUME_FROM):
    model, ckpt_config = load_checkpoint(RESUME_FROM, device="cuda")
    backbone = model.backbone
    print(f"Resumed from {RESUME_FROM}")
else:
    backbone = ConvNeXt_L_Backbone(**BACKBONE_KWARGS)
    model    = Classifier(backbone, **CLASSIFIER_KWARGS)
    # ── load MAE pretrained encoder weights ───────────────────────────────
    if PRETRAINED_ENCODER and os.path.exists(PRETRAINED_ENCODER):
        state = torch.load(PRETRAINED_ENCODER, map_location="cuda")
        missing, unexpected = backbone.backbone.load_state_dict(state, strict=False)
        print(f"Loaded MAE encoder weights from {PRETRAINED_ENCODER}")
        print(f"  missing: {len(missing)}  unexpected: {len(unexpected)}")
    elif PRETRAINED_ENCODER:
        print(f"  Warning: pretrained encoder not found at {PRETRAINED_ENCODER}, training from ImageNet init")

unfreeze_groups = [
    backbone.backbone.stages[3],
    backbone.backbone.stages[2],
    backbone.backbone.stages[1],
    backbone.backbone.stages[0],
    backbone.backbone.stem,
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
    checkpoint_path   = CHECKPOINT,
    model_name        = "ConvNeXt-L",
    backbone_kwargs   = BACKBONE_KWARGS,
    classifier_kwargs = CLASSIFIER_KWARGS,
    fit_kwargs        = FIT_KWARGS,
)

model = fit(model, train_loader, val_loader,
            unfreeze_groups=unfreeze_groups,
            checkpoint_path=CHECKPOINT,
            save_fn=save_fn,
            reporter=reporter,
            resume=False,
            **FIT_KWARGS)