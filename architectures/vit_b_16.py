import torch
import torchvision.models as tvm
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
import torch.nn.functional as F

from architectures.modules import SwiGLU, RMSNorm
from helpers.checkpoint import register

@register
class ViT_B_16_Backbone(nn.Module):
    """
    Pretrained ViT-B/16 adapted for grayscale input
    outputs embedding vector (embed_dim) and alpha (scalar)
    """
    def __init__(self, embed_dim=256, freeze_until='encoder_layer_10', dropout=0.1, finetune_input=True):
        super().__init__()
        base = tvm.vit_b_16(weights=tvm.ViT_B_16_Weights.IMAGENET1K_SWAG_E2E_V1) #384x384
        
        # Adapt patch embedding for grayscale
        old_proj = base.conv_proj
        new_proj = nn.Conv2d(1, old_proj.out_channels, 
                             kernel_size=old_proj.kernel_size, 
                             stride=old_proj.stride, 
                             padding=old_proj.padding, 
                             bias=False)
        new_proj.weight.data = old_proj.weight.data.mean(dim=1, keepdim=True)
        base.conv_proj = new_proj
        
        self.backbone = base
        hidden_dim = base.hidden_dim  # ViT-B/16 hidden dim is 768
        
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, embed_dim),
            nn.GELU(),
            #SwiGLU(embed_dim, hidden_ratio=4/3),
            RMSNorm(embed_dim),
            nn.Dropout(dropout)   
        )
        
        self.alpha_spatial = nn.Linear(hidden_dim, 1)  # scoring each image patch
        self.alpha_head    = nn.Linear(hidden_dim, 1)  # mapping attended context to a scalar
        
        freeze_until_idx = {
            'encoder_layer_0': 0, 'encoder_layer_2': 2, 'encoder_layer_4': 4,
            'encoder_layer_6': 6, 'encoder_layer_8': 8, 'encoder_layer_10': 10,
            'encoder_layer_11': 11, 'encoder_layer_12': 12
        }.get(freeze_until, 12)
        
        start_idx = 0 if not finetune_input else 1
        if start_idx == 0:
            for p in base.conv_proj.parameters():
                p.requires_grad = False
        
        for i, layer in enumerate(base.encoder.layers):
            if i < freeze_until_idx:
                for p in layer.parameters():
                    p.requires_grad = False
                    
        base.encoder.pos_embedding.requires_grad = False
    
    def forward(self, x):
        n = x.shape[0]
        x = self.backbone._process_input(x)

        # Add class token and positional embeddings
        x = torch.cat([self.backbone.class_token.expand(n, 1, -1), x], dim=1)
        x = x + self.backbone.encoder.pos_embedding
        x = self.backbone.encoder.dropout(x)
        
        x = self.backbone.encoder.ln(self.backbone.encoder.layers(x)) # faster
        
        #for layer in self.backbone.encoder.layers: # less vram
        #    x = checkpoint(layer, x, use_reentrant=False)
        #x = self.backbone.encoder.ln(x
        
        cls_token = x[:, 0]           # (N, hidden_dim)
        embed = self.proj(cls_token)  # for classification
        
        spatial  = x[:, 1:]                                        # (N, num_patches, hidden_dim)
        scores   = F.softmax(self.alpha_spatial(spatial), dim=1)   # (N, num_patches, 1)
        context  = (scores * spatial).sum(dim=1)                   # (N, hidden_dim)
        alpha    = self.alpha_head(context)                        # (N, 1)
        
        return alpha, embed