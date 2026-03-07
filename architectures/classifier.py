import torch
import torch.nn as nn
import torch.nn.functional as F
from architectures.modules import RMSNorm

CATEGORIES = ["XR_SHOULDER", "XR_HUMERUS", "XR_ELBOW",
              "XR_FOREARM", "XR_WRIST", "XR_HAND", "XR_FINGER"]

class Classifier(nn.Module):
    def __init__(self, backbone, categories=CATEGORIES, embed_dim=256,
                 dropout=0.1, attn_heads=4, **kwargs):
        super().__init__()
        self.backbone = backbone

        self.inter_attn = nn.ModuleDict({
            cat: nn.MultiheadAttention(embed_dim, num_heads=attn_heads,
                                       batch_first=True, dropout=dropout)
            for cat in categories
        })
        self.inter_ln = nn.ModuleDict({
            cat: RMSNorm(embed_dim) for cat in categories
        })
        self.weight_head = nn.ModuleDict({
            cat: nn.Linear(embed_dim, 1) for cat in categories
        })
        self.classifiers = nn.ModuleDict({
            cat: nn.Sequential(nn.Dropout(dropout), nn.Linear(embed_dim, 1))
            for cat in categories
        })

    def forward(self, image_list, categories):
        logits, all_weights = [], []

        for images, category in zip(image_list, categories):
            images = images.to(next(self.parameters()).device)
            context, embeds = self.backbone(images)              # (N, embed_dim)

            seq = context.unsqueeze(0)                           # (1, N, embed_dim)
            attended, _ = self.inter_attn[category](seq, seq, seq)
            attended = self.inter_ln[category](attended.squeeze(0))  # (N, embed_dim)

            weights = F.softmax(self.weight_head[category](attended), dim=0)  # (N, 1)
            fused   = (weights * embeds).sum(dim=0, keepdim=True)             # (1, embed_dim)

            logit = self.classifiers[category](fused).squeeze()
            logits.append(logit)
            all_weights.append(weights.squeeze(1))

        return torch.stack(logits), all_weights