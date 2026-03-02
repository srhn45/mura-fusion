import torch
import torch.nn as nn
import torch.nn.functional as F

from architectures.modules import SwiGLU

CATEGORIES = ["XR_SHOULDER", "XR_HUMERUS", "XR_ELBOW",
              "XR_FOREARM", "XR_WRIST", "XR_HAND", "XR_FINGER"]

class Classifier(nn.Module):
    def __init__(self, backbone, categories=CATEGORIES, embed_dim=256, swi_ratio=8/3, mlp_depth=1, dropout=0.1):
        super().__init__()
        self.backbone = backbone
        self.classifiers = nn.ModuleDict({
            cat: self._make_head(embed_dim, swi_ratio, mlp_depth, dropout)
            for cat in categories
        })

    def _make_head(self, embed_dim, swi_ratio, mlp_depth, dropout):
        layers = []
        #for _ in range(mlp_depth):
            #layers.extend([
            #    SwiGLU(embed_dim, hidden_ratio=swi_ratio),
            #    nn.RMSNorm(embed_dim),
            #    nn.Dropout(dropout)
            #])
        layers.append(nn.Linear(embed_dim, 1))
        return nn.Sequential(*layers)

    def forward(self, image_list, categories):
        logits, all_weights = [], []
        for images, category in zip(image_list, categories):
            images  = images.to(next(self.parameters()).device)
            alphas, embeds = self.backbone(images)
            weights = F.softmax(alphas, dim=0)
            fused   = (weights * embeds).sum(dim=0, keepdim=True)
            logit   = self.classifiers[category](fused).squeeze()
            logits.append(logit)
            all_weights.append(weights.squeeze(1))
        return torch.stack(logits), all_weights