import torch
import torch.nn as nn
import torch.nn.functional as F
from architectures.modules import RMSNorm

CATEGORIES = ["XR_SHOULDER", "XR_HUMERUS", "XR_ELBOW",
              "XR_FOREARM", "XR_WRIST", "XR_HAND", "XR_FINGER"]

class Classifier(nn.Module):
    def __init__(self, backbone, categories=CATEGORIES, embed_dim=256,
                 dropout=0.1, **kwargs):
        super().__init__()
        self.backbone    = backbone
        self.classifiers = nn.ModuleDict({
            cat: nn.Sequential(nn.Dropout(dropout), nn.Linear(embed_dim, 1))
            for cat in categories
        })

    def forward(self, image_list, categories):
        logits, all_weights = [], []
        for images, category in zip(image_list, categories):
            images = (torch.stack(images) if isinstance(images, list) else
                      images).to(next(self.parameters()).device)
            context, embeds = self.backbone(images)          # (N, embed_dim)
            N = embeds.shape[0]
            weights = torch.ones(N, device=embeds.device) / N
            fused   = embeds.mean(dim=0, keepdim=True)       # (1, embed_dim)
            logit   = self.classifiers[category](fused).squeeze()
            logits.append(logit)
            all_weights.append(weights)
        return torch.stack(logits), all_weights