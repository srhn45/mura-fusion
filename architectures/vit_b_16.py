import torch
import torchvision.models as tvm
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
import torch.nn.functional as F

from architectures.modules import SwiGLU, RMSNorm
from helpers.checkpoint import register

@register
class ViT_B_16_Backbone(nn.Module):
    def __init__(
        self,
        embed_dim=256,
        freeze_until='encoder_layer_0',
        dropout=0.1,
        finetune_input=True,
    ):
        super().__init__()

        base = tvm.vit_b_16(
            weights=tvm.ViT_B_16_Weights.IMAGENET1K_SWAG_E2E_V1
        ) # 384x384

        # --- Adapt patch embedding for grayscale ---
        old_proj = base.conv_proj
        new_proj = nn.Conv2d(
            1,
            old_proj.out_channels,
            kernel_size=old_proj.kernel_size,
            stride=old_proj.stride,
            padding=old_proj.padding,
            bias=False,
        )
        new_proj.weight.data = old_proj.weight.data.mean(dim=1, keepdim=True)
        base.conv_proj = new_proj

        self.backbone = base
        self.hidden_dim = base.hidden_dim  # 768 for ViT-B/16

        # --- CLS projection ---
        self.proj = nn.Sequential(
            nn.Linear(self.hidden_dim, embed_dim),
            nn.GELU(),
            RMSNorm(embed_dim),
            nn.Dropout(dropout),
        )

        # --- Patch-attended projection ---
        self.proj2 = nn.Sequential(
            nn.Linear(self.hidden_dim, embed_dim),
            nn.GELU(),
            RMSNorm(embed_dim),
            nn.Dropout(dropout),
        )

        self.alpha_spatial = nn.Linear(self.hidden_dim, 1)

        # --- Freezing logic ---
        freeze_until_idx = {
            'encoder_layer_0': 0,
            'encoder_layer_2': 2,
            'encoder_layer_4': 4,
            'encoder_layer_6': 6,
            'encoder_layer_8': 8,
            'encoder_layer_10': 10,
            'encoder_layer_12': 12,
        }.get(freeze_until, 12)

        if not finetune_input:
            for p in base.conv_proj.parameters():
                p.requires_grad = False

        for i, layer in enumerate(base.encoder.layers):
            if i < freeze_until_idx:
                for p in layer.parameters():
                    p.requires_grad = False

    def forward(self, x):

        n = x.shape[0]

        x = self.backbone._process_input(x)

        x = torch.cat(
            [self.backbone.class_token.expand(n, 1, -1), x], dim=1
        )
        x = x + self.backbone.encoder.pos_embedding
        x = self.backbone.encoder.dropout(x)

        #for layer in self.backbone.encoder.layers:
            #if self.training and torch.rand(1).item() < 0.1:
                #continue  # skip this layer entirely during training
            # x = checkpoint(layer, x, use_reentrant=False)
            # x = layer(x)
        # x = self.backbone.encoder.ln(x)
        
        x = self.backbone.encoder.ln(self.backbone.encoder.layers(x)) # normal

        

        # --- CLS embedding ---
        cls_token = x[:, 0]
        embed = self.proj(cls_token)

        # --- Patch attention pooling ---
        spatial = x[:, 1:]  # (N, S, hidden_dim)
        scores = F.softmax(self.alpha_spatial(spatial), dim=1)
        context = (scores * spatial).sum(dim=1)
        context = self.proj2(context)

        return context, embed