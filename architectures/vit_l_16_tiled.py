import torch
import torchvision.models as tvm
import torch.nn as nn
import torch.nn.functional as F
from architectures.modules import RMSNorm
from helpers.checkpoint import register

@register
class ViT_L_16_Backbone(nn.Module):
    def __init__(self, embed_dim=256, freeze_until='encoder_layer_22',
                 dropout=0.1, finetune_input=True, tiled=True):
        super().__init__()
        base = tvm.vit_l_16(weights=tvm.ViT_L_16_Weights.DEFAULT)

        # Adapt patch embedding for grayscale
        old_proj = base.conv_proj
        new_proj = nn.Conv2d(1, old_proj.out_channels,
                             kernel_size=old_proj.kernel_size,
                             stride=old_proj.stride,
                             padding=old_proj.padding,
                             bias=False)
        new_proj.weight.data = old_proj.weight.data.mean(dim=1, keepdim=True)
        base.conv_proj = new_proj

        self.backbone   = base
        self.hidden_dim = base.hidden_dim  # 1024
        self.tiled      = tiled

        self.proj = nn.Sequential(
            nn.Linear(self.hidden_dim, embed_dim), nn.GELU(),
            RMSNorm(embed_dim), nn.Dropout(dropout)
        )
        self.proj2 = nn.Sequential(
            nn.Linear(self.hidden_dim, embed_dim), nn.GELU(),
            RMSNorm(embed_dim), nn.Dropout(dropout)
        )
        self.attn_u = nn.Linear(self.hidden_dim, embed_dim, bias=False)
        self.attn_v = nn.Linear(self.hidden_dim, embed_dim, bias=False)
        self.attn_w = nn.Linear(embed_dim, 1, bias=False)

        # Progressive freeze
        freeze_until_idx = {
            'encoder_layer_0': 0,  'encoder_layer_2': 2,  'encoder_layer_4': 4,
            'encoder_layer_6': 6,  'encoder_layer_8': 8,  'encoder_layer_10': 10,
            'encoder_layer_12': 12,'encoder_layer_14': 14,'encoder_layer_16': 16,
            'encoder_layer_18': 18,'encoder_layer_20': 20,'encoder_layer_22': 22,
            'encoder_layer_24': 24
        }.get(freeze_until, 24)

        if not finetune_input:
            for p in base.conv_proj.parameters():
                p.requires_grad = False

        for i, layer in enumerate(base.encoder.layers):
            if i < freeze_until_idx:
                for p in layer.parameters():
                    p.requires_grad = False

        for layer in base.encoder.layers:
            layer.self_attention.dropout = dropout

    def _tile(self, x):
        N, C, H, W = x.shape
        tl = x2[:, :, :H//2,  :W//2 ]
        tr = x2[:, :, :H//2,  W//2: ]
        bl = x2[:, :, H//2:,  :W//2 ]
        br = x2[:, :, H//2:,  W//2: ]
        return torch.stack([tl, tr, bl, br], dim=1).reshape(N * 4, C, H, W)

    def _encode(self, x):
        n = x.shape[0]
        x = self.backbone._process_input(x)
        x = torch.cat([self.backbone.class_token.expand(n, 1, -1), x], dim=1)
        x = x + self.backbone.encoder.pos_embedding
        x = self.backbone.encoder.dropout(x)
        x = self.backbone.encoder.ln(self.backbone.encoder.layers(x))

        cls_token = x[:, 0]
        embed     = self.proj(cls_token)               # (N, embed_dim) — CLS

        spatial = x[:, 1:]                             # (N, S, 1024)
        scores  = self.attn_w(
            torch.tanh(self.attn_v(spatial)) * torch.sigmoid(self.attn_u(spatial))
        )                                              # (N, S, 1)
        context = (F.softmax(scores, dim=1) * spatial).sum(dim=1)  # (N, 1024)
        context = self.proj2(context)                  # (N, embed_dim)
        return context, embed

    def forward(self, x):
        """
        tiled=False (context, embed) each (N, embed_dim)
        tiled=True (context, embed) each (N*4, embed_dim)
        """
        if self.tiled:
            x = self._tile(x)            # (N*4, 1, H, W)
        return self._encode(x)