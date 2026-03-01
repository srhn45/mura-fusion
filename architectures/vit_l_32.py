import torch
import torchvision.models as tvm
import torch.nn as nn

from architectures.modules import SwiGLU

class ViT_L_32_Backbone(nn.Module):
    """
    Pretrained ViT-L/32 adapted for grayscale input
    outputs embedding vector (embed_dim) and alpha (scalar)
    """
    def __init__(self, embed_dim=256, freeze_until='encoder_layer_22', dropout=0.1, finetune_input=True):
        super().__init__()
        base = tvm.vit_l_32(weights=tvm.ViT_L_32_Weights.DEFAULT)
        
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
        hidden_dim = base.hidden_dim  # ViT-L/32 hidden dim is 1024
        
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, embed_dim),
            nn.RMSNorm(embed_dim),
            nn.Dropout(dropout)
        )
        self.alpha_head = nn.Sequential(
            SwiGLU(hidden_dim),
            nn.Linear(hidden_dim, 1)
        )
        
        freeze_until_idx = {
            'encoder_layer_0': 0, 'encoder_layer_2': 2, 'encoder_layer_4': 4,
            'encoder_layer_6': 6, 'encoder_layer_8': 8, 'encoder_layer_10': 10,
            'encoder_layer_12': 12, 'encoder_layer_14': 14, 'encoder_layer_16': 16,
            'encoder_layer_18': 18, 'encoder_layer_20': 20, 'encoder_layer_22': 22
        }.get(freeze_until, 22)
        
        start_idx = 0 if not finetune_input else 1
        if start_idx == 0:
            for p in base.conv_proj.parameters():
                p.requires_grad = False
        
        for i, layer in enumerate(base.encoder.layers):
            if i < freeze_until_idx:
                for p in layer.parameters():
                    p.requires_grad = False
    
    def forward(self, x):
        n = x.shape[0]
        x = self.backbone._process_input(x)

        x = torch.cat([self.backbone.class_token.expand(n, 1, -1), x], dim=1)
        x = x + self.backbone.encoder.pos_embedding
        x = self.backbone.encoder.dropout(x)
        x = self.backbone.encoder.ln(self.backbone.encoder.layers(x))
        cls_token = x[:, 0]
        embed = self.proj(cls_token)       # for classification
        alpha = self.alpha_head(cls_token) # alpha directly from cls_token
        return alpha, embed