import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class ImageEncoder(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()
        self.resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, latent_dim)
    def forward(self, img): return self.resnet(img)

class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, num_heads=num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, dim * 4), nn.SiLU(), nn.Linear(dim * 4, dim))
    def forward(self, x, context):
        attn_out, _ = self.attn(x, context, context)
        x = self.norm1(x + attn_out)
        x = self.norm2(x + self.mlp(x))
        return x

class RedDimensionTransformer(nn.Module):
    def __init__(self, latent_dim=128, hidden_dim=1024, num_layers=12):
        super().__init__()
        self.register_buffer("freqs", torch.pow(2, torch.linspace(0, 9, 10)))
        self.input_proj = nn.Linear(60, hidden_dim)
        self.latent_proj = nn.Linear(latent_dim, hidden_dim)
        self.blocks = nn.ModuleList([TransformerBlock(hidden_dim) for _ in range(num_layers)])
        self.sdf_head = nn.Linear(hidden_dim, 1)

    def forward(self, x, latent):
        b, n, _ = x.shape
        args = x.unsqueeze(-1) * self.freqs * 3.14159
        enc = torch.cat([torch.sin(args), torch.cos(args)], dim=-1).view(b, n, -1)
        h = self.input_proj(enc)
        context = self.latent_proj(latent).unsqueeze(1)
        for block in self.blocks: h = block(h, context)
        return self.sdf_head(h)
