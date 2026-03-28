import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from architectures.modules import RMSNorm
from helpers.checkpoint import register

@register
class ConvNeXt_L_Backbone(nn.Module):
    def __init__(self, embed_dim=256, freeze_until='stage3', dropout=0.1,
                 finetune_input=True, tiled=False):
        super().__init__()
        base = timm.create_model(
            'convnext_large.fb_in22k_ft_in1k_384',
            pretrained=True,
            num_classes=0,
            global_pool=''
        )
        # Adapt stem conv for grayscale
        old_conv = base.stem[0]
        new_conv = nn.Conv2d(1, old_conv.out_channels,
                             kernel_size=old_conv.kernel_size,
                             stride=old_conv.stride,
                             padding=old_conv.padding,
                             bias=old_conv.bias is not None)
        new_conv.weight.data = old_conv.weight.data.mean(dim=1, keepdim=True)
        if old_conv.bias is not None:
            new_conv.bias.data = old_conv.bias.data.clone()
        base.stem[0] = new_conv
        self.backbone  = base
        self.tiled     = tiled

        hidden_dim = 1536
        self.attn_u = nn.Linear(hidden_dim, embed_dim, bias=False)
        self.attn_v = nn.Linear(hidden_dim, embed_dim, bias=False)
        self.attn_w = nn.Linear(embed_dim, 1, bias=False)
        self.proj   = nn.Sequential(
            nn.Linear(hidden_dim, embed_dim), nn.GELU(),
            RMSNorm(embed_dim), nn.Dropout(dropout)
        )

    def _encode(self, x):
        feats   = self.backbone(x)                          # (N, 1536, H, W)
        spatial = feats.flatten(2).transpose(1, 2)          # (N, H*W, 1536)
        scores  = self.attn_w(
            torch.tanh(self.attn_v(spatial)) * torch.sigmoid(self.attn_u(spatial))
        )                                                   # (N, H*W, 1)
        weights = F.softmax(scores, dim=1)
        pooled  = (weights * spatial).sum(dim=1)            # (N, 1536)
        return self.proj(pooled)                            # (N, embed_dim)

    def _tile(self, x):
        N, C, H, W = x.shape
        tl = x2[:, :, :H//2,  :W//2 ]
        tr = x2[:, :, :H//2,  W//2: ]
        bl = x2[:, :, H//2:,  :W//2 ]
        br = x2[:, :, H//2:,  W//2: ]

        # [img0_tl, img0_tr, img0_bl, img0_br, img1_tl, ...]
        return torch.stack([tl, tr, bl, br], dim=1).reshape(N * 4, C, H, W)

    def forward(self, x):
        """
        tiled=False returns (out, out) shape (N, embed_dim) — standard
        tiled=True returns (out, out) shape (N*4, embed_dim) — 4 crops each
        """
        
        if self.tiled:
            x = self._tile(x)       # (N*4, 1, H, W)
        out = self._encode(x)       # (N or N*4, embed_dim)
        return out, out