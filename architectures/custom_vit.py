class TransformerBlock(nn.Module):
    def __init__(self, dim, heads, swiglu_ratio=8/3):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads
        assert self.head_dim * heads == dim

        self.norm1 = nn.RMSNorm(dim)

        self.qkv = nn.Linear(dim, 3 * dim, bias=False)

        self.rope = RoPEPositionalEmbedding(
            dim=self.head_dim,
            max_seq_len=4096
        )

        self.norm2 = nn.RMSNorm(dim)
        self.swiglu = SwiGLU(dim, hidden_ratio=swiglu_ratio)

    def forward(self, x):
        B, N, D = x.shape

        h = self.norm1(x)
        qkv = self.qkv(h)                     # (B, N, 3D)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(B, N, self.heads, self.head_dim)
        k = k.view(B, N, self.heads, self.head_dim)
        v = v.view(B, N, self.heads, self.head_dim)

        # RoPE applied only to queries and keys
        q = self.rope(q)
        k = self.rope(k)

        # attention
        attn = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn = attn.softmax(dim=-1)

        out = attn @ v                        # (B, N, H, Hd)
        out = out.reshape(B, N, D)

        x = x + out
        x = x + self.swiglu(self.norm2(x))
        return x
    
class AttentionPooling(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.kv = nn.Linear(embed_dim, embed_dim, bias=False) 
        self.scale = embed_dim ** -0.5
        
    def forward(self, x):
        B, N, E = x.shape # x: (B, N, E)
        kv = self.kv(x)
        
        query = self.query.expand(B, -1, -1) # (1, 1, E) -> (B, 1, E)
        
        attn_weights = torch.matmul(query, kv.transpose(-2, -1)) * self.scale # (B, 1, N)
        attn_weights = F.softmax(attn_weights, dim=-1)  # (B, 1, N)
        
        pooled = torch.matmul(attn_weights, x)  # (B, 1, E)
        
        return pooled.squeeze(1)  # (B, E)
    
class Custom_ViT(nn.Module):
    def __init__(
        self,
        img_size=256,
        patch_size=16,
        in_chans=1, # grayscale
        output_dim=256,
        embed_dim=256,
        depth=8,
        heads=8,
        head_depth=1,
        dropout=0.1
    ):
        super().__init__()

        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        n_patches = self.patch_embed.n_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        self.blocks = nn.ModuleList([
            nn.Sequential(
                TransformerBlock(embed_dim, heads),
                nn.Dropout(dropout)
            ) for _ in range(depth)
        ])
        
        self.attn_pool = AttentionPooling(embed_dim)

        self.head = nn.ModuleList([
            nn.Sequential(
                SwiGLU(embed_dim), 
                nn.RMSNorm(embed_dim),
                nn.Dropout(dropout)
            ) for _ in range(head_depth)
        ])
        self.out = nn.Linear(embed_dim,  1)
        

    def forward(self, x):
        B = x.size(0)

        x = self.patch_embed(x)  # (B, N, E)

        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls, x), dim=1)  # (B, N+1, E)

        for blk in self.blocks:
            x = blk(x)
            
        pooled = self.attn_pool(x)  # (B, E)

        out = pooled
        for layer in self.head:
            out = layer(out)
        
        alpha = self.out(out)  # (B, 1)
        
        return alpha, pooled