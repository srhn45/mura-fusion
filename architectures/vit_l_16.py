import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from torch.utils.checkpoint import checkpoint
from architectures.modules import RMSNorm
from helpers.checkpoint import register

@register
class ViT_L_16_Backbone(nn.Module):
    """
    At size=518: 37×37 = 1369 patch tokens (14px granularity)
    At size=336: 24×24 = 576  patch tokens (14px granularity)
    At size=224: 16×16 = 256  patch tokens (14px granularity)
    """
    def __init__(self, embed_dim=256, freeze_until='block_20',
                 dropout=0.1, finetune_input=True, use_checkpoint=True):
        super().__init__()

        base = timm.create_model(
            'vit_large_patch14_dinov2',
            pretrained=True,
            num_classes=0,
            global_pool='',       # return all tokens, not pooled
            img_size=336,         # change to 518 if VRAM allows
        )

        # Adapt patch embedding for grayscale
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

        self.backbone       = base
        self.hidden_dim     = base.embed_dim   # 1024 for ViT-L
        self.use_checkpoint = use_checkpoint

        # CLS projection (global summary)
        self.proj_cls = nn.Sequential(
            nn.Linear(self.hidden_dim, embed_dim),
            nn.GELU(), RMSNorm(embed_dim), nn.Dropout(dropout)
        )

        # Patch projection (local tokens → MIL instances)
        self.proj_patch = nn.Sequential(
            nn.Linear(self.hidden_dim, embed_dim),
            nn.GELU(), RMSNorm(embed_dim), nn.Dropout(dropout)
        )

        # Gated MIL attention over patch tokens
        self.attn_v = nn.Linear(embed_dim, embed_dim, bias=False)
        self.attn_u = nn.Linear(embed_dim, embed_dim, bias=False)
        self.attn_w = nn.Linear(embed_dim, 1, bias=False)

        # Progressive freezing — DINOv2-L has 24 blocks
        freeze_map = {f'block_{i}': i for i in range(25)}
        freeze_map['none'] = 0
        freeze_map['all']  = 24
        freeze_until_idx   = freeze_map.get(freeze_until, 20)

        if not finetune_input:
            for p in base.patch_embed.parameters():
                p.requires_grad = False

        for i, block in enumerate(base.blocks):
            if i < freeze_until_idx:
                for p in block.parameters():
                    p.requires_grad = False

    def forward(self, x):
        B = x.shape[0]

        # Patch embedding + position encoding (timm handles this internally)
        tokens = self.backbone.patch_embed(x)               # (B, N, 1024)
        tokens = self.backbone.pos_drop(
            tokens + self.backbone.pos_embed[:, 1:, :]      # skip CLS pos embed
        )
        cls_token = self.backbone.cls_token.expand(B, 1, -1)
        cls_token = cls_token + self.backbone.pos_embed[:, :1, :]
        tokens    = torch.cat([cls_token, tokens], dim=1)   # (B, N+1, 1024)

        # Forward through transformer blocks
        if self.use_checkpoint and self.training:
            for block in self.backbone.blocks:
                tokens = checkpoint(block, tokens, use_reentrant=False)
        else:
            tokens = self.backbone.blocks(tokens)

        tokens = self.backbone.norm(tokens)

        cls     = tokens[:, 0]                              # (B, 1024) global
        patches = tokens[:, 1:]                             # (B, N, 1024) local

        # Project both
        embed   = self.proj_cls(cls)                        # (B, embed_dim)
        spatial = self.proj_patch(patches)                  # (B, N, embed_dim)

        # Gated MIL attention over patch tokens → context vector
        scores  = self.attn_w(
            torch.tanh(self.attn_v(spatial)) *
            torch.sigmoid(self.attn_u(spatial))
        )                                                   # (B, N, 1)
        weights = F.softmax(scores, dim=1)
        context = (weights * spatial).sum(dim=1)            # (B, embed_dim)

        # context: MIL-pooled local evidence  (→ fusion attention in Classifier)
        # embed:   CLS global summary         (→ weighted sum in Classifier)
        return context, embed