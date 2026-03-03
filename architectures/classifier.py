import torch
import torch.nn as nn
import torch.nn.functional as F

from architectures.modules import SwiGLU, RMSNorm

CATEGORIES = ["XR_SHOULDER", "XR_HUMERUS", "XR_ELBOW",
              "XR_FOREARM", "XR_WRIST", "XR_HAND", "XR_FINGER"]

class Classifier(nn.Module):
    def __init__(self, backbone, categories=CATEGORIES, embed_dim=256, swi_ratio=4/3, mlp_depth=1, dropout=0.1):
        super().__init__()
        self.backbone = backbone

        self.category_to_idx = {cat: i for i, cat in enumerate(categories)}
        self.category_embed = nn.Embedding(len(categories), embed_dim)

        self.classifier = nn.Sequential(
            SwiGLU(embed_dim, hidden_ratio=swi_ratio),
            RMSNorm(embed_dim),
            nn.Linear(embed_dim, 1)
        )
        
        self.logit_bias = nn.Parameter(torch.tensor(0.0))

    def forward(self, image_list, categories):
        logits, all_weights = [], []

        for images, category in zip(image_list, categories):
            images = images.to(next(self.parameters()).device)
            alphas, embeds = self.backbone(images)

            weights = F.softmax(alphas, dim=0)
            fused = (weights * embeds).sum(dim=0, keepdim=True)

            cat_idx = torch.tensor(
                [self.category_to_idx[category]],
                device=fused.device
            )

            conditioned = fused + self.category_embed(cat_idx)
            logit = self.classifier(conditioned).squeeze()
            logit = logit + self.logit_bias

            logits.append(logit)
            all_weights.append(weights.squeeze(1))

        return torch.stack(logits), all_weights