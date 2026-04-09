import torch
import torch.nn as nn
import torch.nn.functional as F
from architectures.modules import RMSNorm

CATEGORIES = ["XR_SHOULDER", "XR_HUMERUS", "XR_ELBOW",
              "XR_FOREARM", "XR_WRIST", "XR_HAND", "XR_FINGER"]

class CategoryLoRA(nn.Module):
    """Low-rank adapter: output = base(x) + x @ B @ A, per category."""
    def __init__(self, in_dim, out_dim, categories, rank=8):
        super().__init__()
        self.base = nn.Linear(in_dim, out_dim, bias=True)
        self.lora_B = nn.ParameterDict({
            cat: nn.Parameter(torch.zeros(in_dim, rank)) for cat in categories
        })
        self.lora_A = nn.ParameterDict({
            cat: nn.Parameter(torch.randn(rank, out_dim) * 0.01) for cat in categories
        })

    def forward(self, x, category):
        return self.base(x) + x @ self.lora_B[category] @ self.lora_A[category]


class Classifier(nn.Module):
    def __init__(self, backbone, categories=CATEGORIES, embed_dim=256,
                 dropout=0.1, attn_heads=8, lora_rank=8, **kwargs):
        super().__init__()
        self.backbone    = backbone
        self.inter_attn  = nn.MultiheadAttention(embed_dim, num_heads=attn_heads,
                                                  batch_first=True, dropout=dropout)
        self.inter_ln    = RMSNorm(embed_dim)
        self.weight_head = nn.Linear(embed_dim, 1)
        self.dropout     = nn.Dropout(dropout)
        self.classifier  = CategoryLoRA(embed_dim, 1, categories, rank=lora_rank)

    def forward(self, image_list, categories):
        logits, all_weights = [], []
        for images, category in zip(image_list, categories):
            images = (torch.stack(images) if isinstance(images, list) else
                      images).to(next(self.parameters()).device)
            context, embeds = self.backbone(images)
            seq      = context.unsqueeze(0)
            attended, _ = self.inter_attn(seq, seq, seq)
            attended = self.inter_ln(attended.squeeze(0))
            weights  = F.softmax(self.weight_head(attended), dim=0)
            fused    = (weights * embeds).sum(dim=0, keepdim=True)
            logit    = self.classifier(self.dropout(fused), category).squeeze()
            logits.append(logit)
            all_weights.append(weights.squeeze(1))
        return torch.stack(logits), all_weights