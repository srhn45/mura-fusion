import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from architectures.modules import RMSNorm
from helpers.checkpoint import register

@register
class NFNet_F4_Backbone(nn.Module):
    """
    Pretrained NFNet-F4 adapted for grayscale.
    Trained at 384x384, designed to run at 512x512 (native test resolution).
    Returns (context, embed) matching the ConvNeXt interface.
    ~316M params, 3072-channel output.
    """
    def __init__(self, embed_dim=256, freeze_until='stage3', dropout=0.1, finetune_input=True):
        super().__init__()
        base = timm.create_model(
            'dm_nfnet_f4.dm_in1k',   # ← NFNet-F4, 512x512 test resolution
            pretrained=True,
            num_classes=0,
            global_pool=''
        )
        # Adapt stem for grayscale — NFNet stem first conv is base.stem.conv1
        old_conv = base.stem.conv1
        new_conv = nn.Conv2d(1, old_conv.out_channels,
                             kernel_size=old_conv.kernel_size,
                             stride=old_conv.stride,
                             padding=old_conv.padding,
                             bias=old_conv.bias is not None)
        new_conv.weight.data = old_conv.weight.data.mean(dim=1, keepdim=True)
        if old_conv.bias is not None:
            new_conv.bias.data = old_conv.bias.data.clone()
        base.stem.conv1 = new_conv
        self.backbone = base

        hidden_dim = 3072  # NFNet-F4 output channels (vs 1536 for ConvNeXt-L)
        self.attn_u = nn.Linear(hidden_dim, embed_dim, bias=False)
        self.attn_v = nn.Linear(hidden_dim, embed_dim, bias=False)
        self.attn_w = nn.Linear(embed_dim, 1, bias=False)
        self.proj   = nn.Sequential(
            nn.Linear(hidden_dim, embed_dim), nn.GELU(),
            RMSNorm(embed_dim), nn.Dropout(dropout)
        )

    def forward(self, x):
        feats   = self.backbone(x)                          # (N, 3072, H, W)
        spatial = feats.flatten(2).transpose(1, 2)          # (N, H*W, 3072)
        scores  = self.attn_w(
            torch.tanh(self.attn_v(spatial)) * torch.sigmoid(self.attn_u(spatial))
        )                                                   # (N, H*W, 1)
        weights = F.softmax(scores, dim=1)                  # (N, H*W, 1)
        pooled  = (weights * spatial).sum(dim=1)            # (N, 3072)
        out     = self.proj(pooled)                         # (N, embed_dim)
        return out, out