import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import math

class ProteinDistanceMapEmbedding(nn.Module):
    """Custom embedding layer for protein distance maps"""
    def __init__(self, img_size=64, patch_size=4, in_channels=1, embed_dim=256):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        
        # Patch embedding with learned projection
        self.patch_embed = nn.Sequential(
            nn.Conv2d(in_channels, embed_dim//4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(embed_dim//4),
            nn.GELU(),
            nn.Conv2d(embed_dim//4, embed_dim//2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(embed_dim//2),
            nn.GELU(),
            nn.Conv2d(embed_dim//2, embed_dim, kernel_size=patch_size, stride=patch_size),
            Rearrange('b c h w -> b (h w) c')
        )
        
        # Learnable positional embeddings with distance-aware initialization
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
        self._init_pos_embed()
        
        # CLS token for global representation
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
    def _init_pos_embed(self):
        """Initialize positional embeddings with distance-aware pattern"""
        pos_embed = torch.zeros(1, self.num_patches, self.pos_embed.size(-1))
        
        # Create 2D positional encoding that respects distance map structure
        h = w = int(math.sqrt(self.num_patches))
        for i in range(h):
            for j in range(w):
                idx = i * w + j
                # Encode diagonal distance (important for protein distance maps)
                diagonal_dist = abs(i - j) / max(h, w)
                # Encode absolute position
                pos_x = i / h
                pos_y = j / w
                
                # Create position-aware features
                pos_embed[0, idx, 0::3] = torch.sin(torch.tensor(pos_x * np.pi))
                pos_embed[0, idx, 1::3] = torch.sin(torch.tensor(pos_y * np.pi))
                pos_embed[0, idx, 2::3] = torch.sin(torch.tensor(diagonal_dist * np.pi))
                
        self.pos_embed.data.copy_(pos_embed)
    
    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=B)
        x = torch.cat([cls_tokens, x], dim=1)
        x = x + torch.cat([torch.zeros(B, 1, self.pos_embed.size(-1), device=x.device), 
                          self.pos_embed.expand(B, -1, -1)], dim=1)
        return x

class ProteinAttentionBlock(nn.Module):
    """Attention block with distance-aware bias"""
    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        
        self.norm2 = nn.LayerNorm(embed_dim)
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )
        
        # Distance-aware attention bias (learned)
        self.distance_bias = nn.Parameter(torch.zeros(1, num_heads, 1, 1))
        
    def forward(self, x, mask=None, return_attention=False):
        # Self-attention with residual
        attn_out, attn_weights = self.attn(self.norm1(x), self.norm1(x), self.norm1(x), 
                                         key_padding_mask=mask, need_weights=return_attention)
        x = x + attn_out
        
        # MLP with residual
        x = x + self.mlp(self.norm2(x))
        
        if return_attention:
            return x, attn_weights
        return x

class HybridProteinBlock(nn.Module):
    """Hybrid block combining convolution and attention for local+global features"""
    def __init__(self, embed_dim, num_heads, kernel_size=3):
        super().__init__()
        self.attn_block = ProteinAttentionBlock(embed_dim, num_heads)
        self.embed_dim = embed_dim
        
        # Skip local convolution during masked training - just use attention
        # Local convolution only works with full 16x16 grid
        self.use_local_conv = False
        
    def forward(self, x, mask=None, return_attention=False):
        # For now, just use attention block
        # Local convolution requires fixed spatial structure which breaks with masking
        if return_attention:
            return self.attn_block(x, mask, return_attention=True)
        return self.attn_block(x, mask)

class ProteinMAEEncoder(nn.Module):
    """Encoder specifically designed for protein distance maps"""
    def __init__(self, img_size=64, patch_size=4, embed_dim=256, depth=8, num_heads=8):
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_embed = ProteinDistanceMapEmbedding(img_size, patch_size, 1, embed_dim)
        
        # Alternating hybrid and standard attention blocks
        self.blocks = nn.ModuleList([
            HybridProteinBlock(embed_dim, num_heads) if i % 2 == 0 
            else ProteinAttentionBlock(embed_dim, num_heads)
            for i in range(depth)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        
    def forward(self, x, mask_ratio=0.75, return_attention=False):
        # Embed patches
        x = self.patch_embed(x)
        B, N, D = x.shape
        
        if return_attention:
            # Store attention weights from the last block
            attention_weights = None
            # Apply transformer blocks, capture attention from the last block
            for i, block in enumerate(self.blocks):
                if i == len(self.blocks) - 1: # Capture attention from the last block
                    x, attention_weights = block(x, return_attention=True)
                else:
                    x = block(x)
            
            x = self.norm(x)
            return x, attention_weights, None, None # Return features, attention, and None for ids

        # Masking strategy: keep CLS token, randomly mask others
        num_patches_to_keep = int((N - 1) * (1 - mask_ratio)) + 1  # +1 for CLS
        
        # Generate random noise for all patches except CLS
        noise = torch.rand(B, N - 1, device=x.device)
        
        # Sort noise to get indices
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        
        # Keep CLS token and selected patches
        ids_keep = ids_shuffle[:, :num_patches_to_keep-1]
        ids_keep = torch.cat([torch.zeros(B, 1, dtype=torch.long, device=x.device), 
                             ids_keep + 1], dim=1)
        
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
        
        # Apply transformer blocks
        for block in self.blocks:
            x_masked = block(x_masked)
            
        x_masked = self.norm(x_masked)
        
        # Return both masked representation and info for reconstruction
        return x_masked, None, ids_restore, ids_keep # Features, Attention, ids_restore, ids_keep

class ProteinMAEDecoder(nn.Module):
    """Decoder with multi-scale reconstruction"""
    def __init__(self, embed_dim=256, decoder_embed_dim=128, decoder_depth=4, 
                 decoder_num_heads=8, patch_size=4):
        super().__init__()
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim)
        
        # Mask token for missing patches
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        
        # Decoder position embeddings
        num_patches = (64 // patch_size) ** 2
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim))
        
        # Decoder blocks
        self.decoder_blocks = nn.ModuleList([
            ProteinAttentionBlock(decoder_embed_dim, decoder_num_heads)
            for _ in range(decoder_depth)
        ])
        
        self.decoder_norm = nn.LayerNorm(decoder_embed_dim)
        
        # Multi-scale reconstruction heads
        self.decoder_pred = nn.ModuleDict({
            'fine': nn.Linear(decoder_embed_dim, patch_size**2),
            'coarse': nn.Sequential(
                nn.Linear(decoder_embed_dim, decoder_embed_dim//2),
                nn.GELU(),
                nn.Linear(decoder_embed_dim//2, (patch_size//2)**2)
            )
        })
        
        self.patch_size = patch_size
        
    def forward(self, x, ids_restore, ids_keep):
        B, N_keep, D = x.shape
        N = ids_restore.shape[1] + 1  # +1 for CLS token
        
        # Embed encoded tokens
        x = self.decoder_embed(x)
        
        # Prepare mask tokens
        mask_tokens = self.mask_token.repeat(B, N - N_keep, 1)
        
        # Combine visible and mask tokens (excluding CLS for now)
        x_no_cls = torch.cat([x[:, 1:], mask_tokens], dim=1)
        
        # Unshuffle
        x_no_cls = torch.gather(x_no_cls, dim=1, 
                               index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))
        
        # Add back CLS token
        x = torch.cat([x[:, :1], x_no_cls], dim=1)
        
        # Add position embeddings
        x = x + self.decoder_pos_embed
        
        # Apply decoder blocks
        for block in self.decoder_blocks:
            x = block(x)
            
        x = self.decoder_norm(x)
        
        # Multi-scale predictions (remove CLS token)
        x_patches = x[:, 1:]
        
        # Fine-scale reconstruction
        fine_pred = self.decoder_pred['fine'](x_patches)
        fine_pred = rearrange(fine_pred, 'b (h w) (p1 p2) -> b 1 (h p1) (w p2)', 
                             h=16, w=16, p1=self.patch_size, p2=self.patch_size)
        
        # Coarse-scale reconstruction (useful for training stability)
        coarse_pred = self.decoder_pred['coarse'](x_patches)
        coarse_pred = rearrange(coarse_pred, 'b (h w) (p1 p2) -> b 1 (h p1) (w p2)', 
                               h=16, w=16, p1=self.patch_size//2, p2=self.patch_size//2)
        coarse_pred = F.interpolate(coarse_pred, size=(64, 64), mode='bilinear', align_corners=False)
        
        # Combine predictions with learnable weighting
        combined_pred = torch.sigmoid(fine_pred)
        
        return combined_pred, fine_pred, coarse_pred

class ProteinDistanceMAE(nn.Module):
    """Complete Masked Autoencoder for Protein Distance Maps"""
    def __init__(self, img_size=64, patch_size=4, embed_dim=256, decoder_embed_dim=128,
                 depth=8, decoder_depth=4, num_heads=8, decoder_num_heads=8):
        super().__init__()
        
        self.encoder = ProteinMAEEncoder(img_size, patch_size, embed_dim, depth, num_heads)
        self.decoder = ProteinMAEDecoder(embed_dim, decoder_embed_dim, decoder_depth, 
                                        decoder_num_heads, patch_size)
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
                
    def forward(self, x, mask_ratio=0.75):
        latent, mask, ids_restore, ids_keep = self.encoder(x, mask_ratio)
        if mask_ratio == 0.0:
            # If no masking, ids_restore is None, so create a tensor of indices
            # that restores the original order of the patches
            num_patches = (self.encoder.patch_embed.img_size // self.encoder.patch_embed.patch_size) ** 2
            ids_restore = torch.arange(num_patches, device=latent.device).unsqueeze(0)
        pred, fine_pred, coarse_pred = self.decoder(latent, ids_restore, ids_keep)
        return pred, fine_pred, coarse_pred, mask
    
    def forward_loss(self, x, pred, ids_keep):
        """
        Compute reconstruction loss with focus on visible patches
        """
        target = x
        
        # Simple MSE loss
        loss = F.mse_loss(pred, target, reduction='none')
        
        # Optionally weight the loss to focus more on unmasked regions during early training
        # This can help the model learn better representations
        loss = loss.mean()
        
        return loss

class ProteinMAELoss(nn.Module):
    """Custom loss function for protein distance maps"""
    def __init__(self, smoothness_weight=0.1, symmetry_weight=0.1):
        super().__init__()
        self.smoothness_weight = smoothness_weight
        self.symmetry_weight = symmetry_weight
        
    def forward(self, pred, target):
        # Reconstruction loss
        recon_loss = F.mse_loss(pred, target)
        
        # Smoothness loss (distance maps should be locally smooth)
        dx = torch.abs(pred[:, :, 1:, :] - pred[:, :, :-1, :])
        dy = torch.abs(pred[:, :, :, 1:] - pred[:, :, :, :-1])
        smoothness_loss = (dx.mean() + dy.mean()) * self.smoothness_weight
        
        # Symmetry loss (distance maps should be symmetric)
        pred_T = pred.transpose(-2, -1)
        symmetry_loss = F.mse_loss(pred, pred_T) * self.symmetry_weight
        
        total_loss = recon_loss + smoothness_loss + symmetry_loss
        
        return total_loss, {
            'recon': recon_loss.item(),
            'smooth': smoothness_loss.item(),
            'symmetry': symmetry_loss.item()
        }

# Training utilities
def create_optimizer(model, lr=1e-4, weight_decay=0.05):
    """Create AdamW optimizer with proper weight decay"""
    # Separate parameters that should and shouldn't have weight decay
    decay = []
    no_decay = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        
        if len(param.shape) == 1 or name.endswith('.bias') or 'pos_embed' in name:
            no_decay.append(param)
        else:
            decay.append(param)
            
    return torch.optim.AdamW([
        {'params': decay, 'weight_decay': weight_decay},
        {'params': no_decay, 'weight_decay': 0.0}
    ], lr=lr)

def cosine_scheduler(base_value, final_value, epochs, niter_per_ep, warmup_epochs=5):
    """Cosine learning rate scheduler with warmup"""
    warmup_schedule = np.linspace(0, base_value, warmup_epochs * niter_per_ep)
    
    iters = np.arange(epochs * niter_per_ep - warmup_epochs * niter_per_ep)
    schedule = final_value + 0.5 * (base_value - final_value) * \
               (1 + np.cos(np.pi * iters / len(iters)))
    
    schedule = np.concatenate((warmup_schedule, schedule))
    
    return schedule