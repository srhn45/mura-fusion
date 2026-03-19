import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from architectures.modules import RMSNorm
from helpers.checkpoint import register

@register
class EVA02_L_Backbone(nn.Module):
    """
    448x448 native resolution, ~307M params, 1024-dim spatial tokens.
    """
    def __init__(self, embed_dim=256, freeze_until='encoder_layer_0',
                 dropout=0.1, finetune_input=True):
        super().__init__()
        base = timm.create_model(
            'eva02_large_patch14_448.mim_m38m_ft_in22k_in1k',
            pretrained=True,
            num_classes=0,
            global_pool=''   # return full spatial tokens
        )

        # Adapt patch embed for grayscale
        # EVA02 uses base.patch_embed.proj as the stem conv
        old_proj = base.patch_embed.proj
        new_proj = nn.Conv2d(1, old_proj.out_channels,
                             kernel_size=old_proj.kernel_size,
                             stride=old_proj.stride,
                             padding=old_proj.padding,
                             bias=old_proj.bias is not None)
        new_proj.weight.data = old_proj.weight.data.mean(dim=1, keepdim=True)
        if old_proj.bias is not None:
            new_proj.bias.data = old_proj.bias.data.clone()
        base.patch_embed.proj = new_proj
        self.backbone = base

        hidden_dim = 1024  # EVA02-L hidden dim

        # Gated MIL attention over spatial tokens
        self.attn_u = nn.Linear(hidden_dim, embed_dim, bias=False)
        self.attn_v = nn.Linear(hidden_dim, embed_dim, bias=False)
        self.attn_w = nn.Linear(embed_dim, 1, bias=False)

        # CLS token projection
        self.proj  = nn.Sequential(
            nn.Linear(hidden_dim, embed_dim), nn.GELU(),
            RMSNorm(embed_dim), nn.Dropout(dropout)
        )
        # Spatial context projection
        self.proj2 = nn.Sequential(
            nn.Linear(hidden_dim, embed_dim), nn.GELU(),
            RMSNorm(embed_dim), nn.Dropout(dropout)
        )

        # Progressive freeze — EVA02 blocks are in base.blocks
        freeze_map = {f'encoder_layer_{i}': i for i in range(0, 25, 2)}
        freeze_map['none'] = len(base.blocks)
        freeze_until_idx = freeze_map.get(freeze_until, 0)

        if not finetune_input:
            for p in base.patch_embed.parameters():
                p.requires_grad = False

        for i, block in enumerate(base.blocks):
            if i < freeze_until_idx:
                for p in block.parameters():
                    p.requires_grad = False

    def forward(self, x):
        # EVA02 forward_features returns (N, num_tokens, hidden_dim)
        # num_tokens = 1 (cls) + H*W/patch^2 spatial tokens
        tokens = self.backbone.forward_features(x)   # (N, T, 1024)

        cls_token = tokens[:, 0]                     # (N, 1024)
        spatial   = tokens[:, 1:]                    # (N, S, 1024)

        # gated MIL attention over spatial tokens
        scores  = self.attn_w(
            torch.tanh(self.attn_v(spatial)) * torch.sigmoid(self.attn_u(spatial))
        )                                            # (N, S, 1)
        weights = F.softmax(scores, dim=1)
        context = (weights * spatial).sum(dim=1)     # (N, 1024)
        context = self.proj2(context)                # (N, embed_dim)

        embed = self.proj(cls_token)                 # (N, embed_dim)
        return context, embed