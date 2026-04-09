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
        self.backbone = base
        self.tiled    = tiled
        hidden_dim    = 1536
        self.bottle   = nn.Sequential(
            nn.Linear(hidden_dim, embed_dim), nn.GELU(),
            RMSNorm(embed_dim), nn.Dropout(dropout)
        )
        if self.tiled:
            self.tile_embed = nn.Embedding(4, embed_dim)
        self.attn_u  = nn.Linear(embed_dim, embed_dim, bias=False)
        self.attn_v  = nn.Linear(embed_dim, embed_dim, bias=False)
        self.attn_w  = nn.Linear(embed_dim, 1, bias=False)
        self.out_norm = nn.Sequential(RMSNorm(embed_dim), nn.Dropout(dropout))

    def _encode(self, x):
        feats   = torch.utils.checkpoint.checkpoint(self.backbone, x, use_reentrant=False)
        spatial = feats.flatten(2).transpose(1, 2)
        return self.bottle(spatial)

    def forward(self, x):
        if not self.tiled:
            spatial = self._encode(x)
            scores  = self.attn_w(
                torch.tanh(self.attn_v(spatial)) * torch.sigmoid(self.attn_u(spatial))
            )
            weights = F.softmax(scores, dim=1)
            pooled  = (weights * spatial).sum(dim=1)
            return self.out_norm(pooled), self.out_norm(pooled)

        N, C, H, W = x.shape
        assert H % 2 == 0 and W % 2 == 0
        tiles = [
            x[:, :, :H//2, :W//2], x[:, :, :H//2, W//2:],
            x[:, :, H//2:, :W//2], x[:, :, H//2:, W//2:],
        ]
        spatial = []
        for i, tile in enumerate(tiles):
            enc = self._encode(tile)
            pos = self.tile_embed(
                torch.full((N, enc.shape[1]), i, device=x.device)
            )
            spatial.append(enc + pos)
        spatial = torch.cat(spatial, dim=1)
        scores  = self.attn_w(
            torch.tanh(self.attn_v(spatial)) * torch.sigmoid(self.attn_u(spatial))
        )
        weights = F.softmax(scores, dim=1)
        pooled  = (weights * spatial).sum(dim=1)
        return self.out_norm(pooled), self.out_norm(pooled)


@register
class ConvNeXt_Tiny_MIL_Backbone(nn.Module):
    """
    True patch-level MIL using ConvNeXt-Tiny.

    The input image is divided into a grid of non-overlapping patches.
    Each patch is resized to the backbone's native resolution and processed
    independently — no patch ever sees pixels from another patch, making
    the MIL attention weights valid for localization.

    At grid=4 (default): 16 patches per image, each 1/4 of the input area.
    """
    def __init__(self, embed_dim=256, dropout=0.1,
                 grid=4, patch_size=224, **kwargs):
        super().__init__()
        base = timm.create_model(
            'convnext_tiny.fb_in22k_ft_in1k',
            pretrained=True,
            num_classes=0,
            global_pool='avg',   # pool each patch to a single vector
        )
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

        self.backbone   = base
        self.grid       = grid          # splits per side → grid² patches total
        self.patch_size = patch_size    # resize each patch to this before backbone

        hidden_dim = 768  # ConvNeXt-Tiny output channels
        self.bottle = nn.Sequential(
            nn.Linear(hidden_dim, embed_dim), nn.GELU(),
            RMSNorm(embed_dim), nn.Dropout(dropout)
        )
        # positional embedding — one per grid cell
        self.pos_embed = nn.Embedding(grid * grid, embed_dim)

        self.attn_u  = nn.Linear(embed_dim, embed_dim, bias=False)
        self.attn_v  = nn.Linear(embed_dim, embed_dim, bias=False)
        self.attn_w  = nn.Linear(embed_dim, 1, bias=False)
        self.out_norm = nn.Sequential(RMSNorm(embed_dim), nn.Dropout(dropout))

    def _split_patches(self, x):
        N, C, H, W = x.shape
        g  = self.grid
        ph = H // g
        pw = W // g
        patches = []
        for i in range(g):
            for j in range(g):
                patches.append(x[:, :, i*ph:(i+1)*ph, j*pw:(j+1)*pw])
        return patches   # list of g² tensors, each (N, 1, ph, pw)

    def forward(self, x):
        N = x.shape[0]
        patches = self._split_patches(x)   # g² × (N, 1, ph, pw)

        embeddings = []
        for idx, patch in enumerate(patches):
            # resize each isolated patch to backbone's native resolution
            p = F.interpolate(patch, size=(self.patch_size, self.patch_size),
                              mode='bilinear', align_corners=False)
            feat = torch.utils.checkpoint.checkpoint(
                self.backbone, p, use_reentrant=False
            )                                          # (N, 768)
            feat = self.bottle(feat)                   # (N, embed_dim)
            pos  = self.pos_embed(
                torch.full((N,), idx, dtype=torch.long, device=x.device)
            )                                          # (N, embed_dim)
            embeddings.append(feat + pos)

        # stack: (N, g², embed_dim)
        spatial = torch.stack(embeddings, dim=1)

        # gated MIL attention over g² patch embeddings
        scores  = self.attn_w(
            torch.tanh(self.attn_v(spatial)) * torch.sigmoid(self.attn_u(spatial))
        )                                              # (N, g², 1)
        weights = F.softmax(scores, dim=1)
        pooled  = (weights * spatial).sum(dim=1)       # (N, embed_dim)

        out = self.out_norm(pooled)
        return out, out                                # matches classifier interface