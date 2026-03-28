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
        self.bottle = nn.Sequential(
            nn.Linear(hidden_dim, embed_dim), nn.GELU(),
            RMSNorm(embed_dim), nn.Dropout(dropout)
        )
        
        if self.tiled:
            self.tile_embed = nn.Embedding(4, embed_dim)
        
        self.attn_u = nn.Linear(embed_dim, embed_dim, bias=False)
        self.attn_v = nn.Linear(embed_dim, embed_dim, bias=False)
        self.attn_w = nn.Linear(embed_dim, 1, bias=False)
        
        self.out_norm = nn.Sequential(RMSNorm(embed_dim), nn.Dropout(dropout))

    def _encode(self, x):
        feats = torch.utils.checkpoint.checkpoint(self.backbone, x, use_reentrant=False)
        #feats   = self.backbone(x)                          # (N, 1536, H, W)
        spatial = feats.flatten(2).transpose(1, 2)          # (N, H*W, 1536)
        spatial = self.bottle(spatial)
        return spatial

    def forward(self, x):
        if not self.tiled:
            spatial = self._encode(x)

            scores  = self.attn_w(
                torch.tanh(self.attn_v(spatial)) * torch.sigmoid(self.attn_u(spatial))
            )                                                   # (N, H*W, 1)
            weights = F.softmax(scores, dim=1)
            pooled  = (weights * spatial).sum(dim=1)            # (N, 1536)
            
            output = self.out_norm(pooled)
            return output, output

        N, C, H, W = x.shape
        assert H % 2 == 0 and W % 2 == 0

        tiles = [
            x[:, :, :H//2,  :W//2],  # tl
            x[:, :, :H//2,  W//2:],  # tr
            x[:, :, H//2:,  :W//2],  # bl
            x[:, :, H//2:,  W//2:],  # br
        ]

        spatial = []

        for i, tile in enumerate(tiles):
            enc = self._encode(tile)          # (N, HW_tile, embed_dim)
            pos = self.tile_embed(
                torch.full((N, enc.shape[1]), i, device=x.device)
            )                                 # (N, HW_tile, embed_dim)
            spatial.append(enc + pos)
            

        spatial = torch.cat(spatial, dim=1)    # (N, 4*HW_tile, 1536)
        
        scores  = self.attn_w(
            torch.tanh(self.attn_v(spatial)) * torch.sigmoid(self.attn_u(spatial))
        )                                                   # (N, H*W, 1)
        weights = F.softmax(scores, dim=1)
        pooled  = (weights * spatial).sum(dim=1)            # (N, 1536)
        
        output = self.out_norm(pooled)
        return output, output
