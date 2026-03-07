import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvm
from architectures.modules import SwiGLU, RMSNorm
from helpers.checkpoint import register

@register
class ResNet152_Backbone(nn.Module):
    def __init__(self, embed_dim=256, freeze_until='layer4', dropout=0.1, finetune_input=True):
        super().__init__()
        base = tvm.resnet152(weights=tvm.ResNet152_Weights.DEFAULT)

        old_conv = base.conv1
        new_conv = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        new_conv.weight.data = old_conv.weight.data.mean(dim=1, keepdim=True)
        base.conv1 = new_conv

        self.backbone = nn.Sequential(
            base.conv1, base.bn1, base.relu, base.maxpool,  # indices 0-3
            base.layer1, base.layer2, base.layer3, base.layer4  # indices 4-7
        )

        hidden_dim = 2048

        self.attn_u = nn.Linear(hidden_dim, embed_dim, bias=False)
        self.attn_v = nn.Linear(hidden_dim, embed_dim, bias=False)
        self.attn_w = nn.Linear(embed_dim, 1, bias=False)
        self.proj   = nn.Sequential(
            nn.Linear(hidden_dim, embed_dim), nn.GELU(),
            RMSNorm(embed_dim), nn.Dropout(dropout)
        )

        freeze_to = {
            'layer0': 0,   # ← freeze nothing
            'layer1': 5, 'layer2': 6, 'layer3': 7, 'layer4': 8
        }
        start_freeze_idx = 1 if finetune_input else 0
        for layer in list(self.backbone.children())[start_freeze_idx:freeze_to.get(freeze_until, 8)]:
            for p in layer.parameters():
                p.requires_grad = False

    def forward(self, x):
        feats   = self.backbone(x)                                    # (N, 2048, H, W)
        spatial = feats.flatten(2).transpose(1, 2)                    # (N, H*W, 2048)
        
        scores  = self.attn_w(
            torch.tanh(self.attn_v(spatial)) * torch.sigmoid(self.attn_u(spatial))
        ) # (N, H*W, 1)
        
        weights = F.softmax(scores, dim=1)                            # (N, H*W, 1)
        pooled  = (weights * spatial).sum(dim=1)                      # (N, 2048)
        out     = self.proj(pooled)                                   # (N, embed_dim)
        return out, out