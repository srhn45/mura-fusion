import torch
import torch.nn as nn
import torch.nn.functional as F

from architectures.modules import SwiGLU

class Classifier(nn.Module):
    """
    Multi-image patient-level classifier using attention-based fusion.

    For each patient:
      1. Run backbone model over every image
      2. Softmax over {alpha_i}
      3. Weighted sum of embeds
      4. Classification network
    """
    def __init__(self, backbone, embed_dim=256, swi_ratio=8/3, mlp_depth=1, dropout=0.1):
        super().__init__()
        self.backbone = backbone

        layers = []
        in_dim = embed_dim
        for _ in range(mlp_depth):
            layers.extend([
                SwiGLU(embed_dim, hidden_ratio=swi_ratio),
                nn.RMSNorm(embed_dim),
                nn.Dropout(dropout)
            ])
        layers.append(nn.Linear(embed_dim, 1))

        self.classifier = nn.Sequential(*layers)

    def forward(self, image_list):
        logits = []
        all_weights = []

        for images in image_list:                          # images: (N, 1, H, W)
            images = images.to(next(self.parameters()).device)
            alphas, embeds = self.backbone(images)         # (N, 1), (N, embed_dim)
            weights = F.softmax(alphas, dim=0)
            fused = (weights * embeds).sum(dim=0, keepdim=True)
            logit = self.classifier(fused).squeeze()
            logits.append(logit)
            all_weights.append(weights.squeeze(1))

        return torch.stack(logits), all_weights  # (B, 1)