import torch
import torchvision.models as tvm
import torch.nn as nn

from architectures.modules import SwiGLU
from helpers.checkpoint import register

@register
class ResNet152_Backbone(nn.Module):
    """
    Pretrained ResNet-152 adapted for grayscale input 
    outputs embedding vector (embed_dim) and alpha (scalar)
    """
    def __init__(self, embed_dim=256, freeze_until='layer3', dropout=0.1, finetune_input=True):
        super().__init__()
        base = tvm.resnet152(weights=tvm.ResNet152_Weights.DEFAULT)

        # Adapt input conv for single-channel grayscale
        old_conv = base.conv1
        new_conv = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        new_conv.weight.data = old_conv.weight.data.mean(dim=1, keepdim=True) # average over the rgb channels
        base.conv1 = new_conv

        self.backbone = nn.Sequential(
            base.conv1, base.bn1, base.relu, base.maxpool,
            base.layer1, base.layer2, base.layer3, base.layer4
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Sequential(
            nn.Linear(2048, embed_dim),  # ResNet-152 also has 2048 output features
            nn.RMSNorm(embed_dim),
            nn.Dropout(dropout)
        )
        self.alpha_head = nn.Sequential(
            SwiGLU(2048),
            nn.Linear(embed_dim, 1)
        )
            
        freeze_to = {'layer1': 5, 'layer2': 6, 'layer3': 7, 'layer4': 8}
        
        if finetune_input: # Start freezing from after conv1 (index 1 onwards)
            start_freeze_idx = 1
        else: # Freeze from the beginning (including conv1)
            start_freeze_idx = 0
            
        layers_to_freeze = list(self.backbone.children())[start_freeze_idx:freeze_to.get(freeze_until, 8)]
        
        for layer in layers_to_freeze:
            for p in layer.parameters():
                p.requires_grad = False

    def forward(self, x):
        feats = self.backbone(x)
        pooled = self.pool(feats).flatten(1)
        embed = self.proj(pooled)
        alpha = self.alpha_head(pooled)
        return alpha, embed