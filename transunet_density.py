import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, c_in, c_out):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(c_in, c_out, 3, padding=1),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(c_out, c_out, 3, padding=1),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
        )
    def forward(self, x): 
        return self.net(x)

class ViTBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, num_layers, spatial_size=32):
        super().__init__()
        self.spatial_size = spatial_size
        num_patches = spatial_size * spatial_size
        
        # Trainable Positional Embedding
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches, embed_dim) * 0.02)
        
        # Using batch_first=True so input shape is [B, SeqLen, EmbedDim]
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=num_heads, 
            dim_feedforward=embed_dim * 4, 
            activation="gelu", 
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
    def forward(self, x):
        # x is a feature map: [B, C, H, W]
        B, C, H, W = x.shape
        # Flatten spatial dimensions into a sequence
        x = x.flatten(2).transpose(1, 2)  # [B, H*W, C]
        
        # Add spatial positional embedding
        x = x + self.pos_embed
        
        # Apply Self-Attention
        x = self.transformer(x)
        
        # Reshape back to feature map
        x = x.transpose(1, 2).view(B, C, H, W)
        return x

class TransUNetDensity(nn.Module):
    """
    Hybrid TransUNet for Density Map Regression.
    Combines a standard CNN Encoder with a Vision Transformer Bottleneck,
    and a standard CNN Decoder with skip connections.
    """
    def __init__(self):
        super().__init__()
        # Encoder
        self.d1 = ConvBlock(1, 32)
        self.d2 = ConvBlock(32, 64)
        self.b  = ConvBlock(64, 128)
        
        # Vision Transformer Bottleneck
        # The feature map here will be 128 channels at 32x32 spatial resolution (1024 tokens)
        self.vit = ViTBlock(embed_dim=128, num_heads=8, num_layers=4, spatial_size=32)

        # Decoder
        self.u2 = ConvBlock(128+64, 64)
        self.u1 = ConvBlock(64+32, 32)
        
        # Output Heads
        self.den_head = nn.Conv2d(32, 1, 1)
        self.msk_head = nn.Conv2d(32, 1, 1)

    def forward(self, x):
        # --- Encoder Path ---
        d1 = self.d1(x)                                  # [B, 32, 128, 128]
        p1 = F.max_pool2d(d1, 2)                         # [B, 32, 64, 64]
        
        d2 = self.d2(p1)                                 # [B, 64, 64, 64]
        p2 = F.max_pool2d(d2, 2)                         # [B, 64, 32, 32]
        
        b  = self.b(p2)                                  # [B, 128, 32, 32]
        
        # --- Transformer Bottleneck ---
        # The self-attention looks at the entire 32x32 feature map globally
        b = self.vit(b)                                  # [B, 128, 32, 32]
        
        # --- Decoder Path ---
        up2 = F.interpolate(b, scale_factor=2, mode="bilinear", align_corners=False) # [B, 128, 64, 64]
        u2 = self.u2(torch.cat([up2, d2], 1))                                        # [B, 64, 64, 64]
        
        up1 = F.interpolate(u2, scale_factor=2, mode="bilinear", align_corners=False)# [B, 64, 128, 128]
        u1 = self.u1(torch.cat([up1, d1], 1))                                        # [B, 32, 128, 128]
        
        # --- Prediction ---
        pred_den = torch.sigmoid(self.den_head(u1))   # [0,1] Density Map
        pred_msk_logit = self.msk_head(u1)            # Raw Mask Logits
        
        return pred_den, pred_msk_logit, b
