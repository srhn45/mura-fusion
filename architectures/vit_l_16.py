import torch
import torchvision.models as tvm
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
import torch.nn.functional as F

from architectures.modules import SwiGLU, RMSNorm
from helpers.checkpoint import register

@register
class ViT_L_16_Backbone(nn.Module):
    """
    Pretrained ViT-L/16 adapted for grayscale input
    outputs embedding vector (embed_dim) and alpha (scalar)
    """
    def __init__(self, embed_dim=256, freeze_until='encoder_layer_22', dropout=0.1, finetune_input=True):
        super().__init__()
        base = tvm.vit_l_16(weights=tvm.ViT_L_16_Weights.DEFAULT) #224x224
        # base = tvm.vit_l_16(weights=tvm.ViT_L_16_Weights.IMAGENET1K_SWAG_E2E_V1) # 512x512
        
        self.dropout = nn.Dropout(dropout)
        
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
        self.hidden_dim = base.hidden_dim  # ViT-L/16 hidden dim is 1024
        
        self.proj = nn.Sequential(
            nn.Linear(self.hidden_dim, embed_dim),
            nn.GELU(),
            RMSNorm(embed_dim),
            #SwiGLU(embed_dim, hidden_ratio=4/3),
            #RMSNorm(embed_dim),
            nn.Dropout(dropout) 
        )
        
        self.proj2 = nn.Sequential(
            nn.Linear(self.hidden_dim, embed_dim),
            nn.GELU(),
            RMSNorm(embed_dim),
            #SwiGLU(embed_dim, hidden_ratio=4/3),
            #RMSNorm(embed_dim),
            nn.Dropout(dropout)  
        )
        
        self.attn_u = nn.Linear(self.hidden_dim, embed_dim, bias=False)
        self.attn_v = nn.Linear(self.hidden_dim, embed_dim, bias=False)
        self.attn_w = nn.Linear(embed_dim, 1, bias=False)
        
        freeze_until_idx = {
            'encoder_layer_0': 0, 'encoder_layer_2': 2, 'encoder_layer_4': 4,
            'encoder_layer_6': 6, 'encoder_layer_8': 8, 'encoder_layer_10': 10,
            'encoder_layer_12': 12, 'encoder_layer_14': 14, 'encoder_layer_16': 16,
            'encoder_layer_18': 18, 'encoder_layer_20': 20, 'encoder_layer_22': 22,
            'encoder_layer_24': 24
        }.get(freeze_until, 24)
        
        start_idx = 0 if not finetune_input else 1
        if start_idx == 0:
            for p in base.conv_proj.parameters():
                p.requires_grad = False
        
        for i, layer in enumerate(base.encoder.layers):
            if i < freeze_until_idx:
                for p in layer.parameters():
                    p.requires_grad = False
        
       
        #for p in base.encoder.ln.parameters():
        #    p.requires_grad = False
        #base.class_token.requires_grad = False
        
        for layer in base.encoder.layers:
            layer.self_attention.dropout = dropout  # was 0.0 in pretrained weights
        
        self.drop_rates = [0.1 * i / len(self.backbone.encoder.layers) 
              for i in range(len(self.backbone.encoder.layers))]
    
    def forward(self, x):
        n = x.shape[0]
        x = self.backbone._process_input(x)

        x = torch.cat([self.backbone.class_token.expand(n, 1, -1), x], dim=1)
        x = x + self.backbone.encoder.pos_embedding
        x = self.backbone.encoder.dropout(x)
        
        x = self.backbone.encoder.ln(self.backbone.encoder.layers(x)) # faster
        
        #for layer in self.backbone.encoder.layers: # less vram
        #    x = checkpoint(layer, x, use_reentrant=False)
        #x = self.backbone.encoder.ln(x)

        #for i, layer in enumerate(self.backbone.encoder.layers):
            #if self.training and torch.rand(1).item() < self.drop_rates[i]:
            #    x = x  # skip residual, keep identity
            #else:
            #    x = checkpoint(layer, x, use_reentrant=False) 
            #x = checkpoint(layer, x, use_reentrant=False) 
        
        cls_token = x[:, 0]
        embed = self.proj(cls_token)       # for classification
        
        spatial  = x[:, 1:]                                        # (N, S, hidden_dim)
        scores  = self.attn_w(
            torch.tanh(self.attn_v(spatial)) * torch.sigmoid(self.attn_u(spatial))
        ) # (N, H*W, 1)
        context  = (scores * spatial).sum(dim=1) # (N, hidden_dim)
        context = self.proj2(context)
        
        return context, embed