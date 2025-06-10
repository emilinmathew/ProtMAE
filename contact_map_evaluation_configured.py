import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import os
import glob
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score, precision_score
import pickle
import random
import math

# Import your MAE model
from custom_Vit import ProteinDistanceMAE

class AdvancedDiceLoss(nn.Module):
    """Dice Loss specifically designed for protein contact prediction"""
    def __init__(self, smooth=1.0, gamma=2.0):
        super().__init__()
        self.smooth = smooth
        self.gamma = gamma
        
    def forward(self, inputs, targets):
        # Convert to probabilities if logits
        if inputs.min() < 0 or inputs.max() > 1:
            inputs = torch.sigmoid(inputs)
        
        # Flatten tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        # Dice coefficient
        intersection = (inputs * targets).sum()
        total = inputs.sum() + targets.sum()
        dice = (2.0 * intersection + self.smooth) / (total + self.smooth)
        
        # Dice loss with gamma for harder examples
        dice_loss = 1 - dice
        return dice_loss ** self.gamma

class CombinedLoss(nn.Module):
    """Combination of Focal Loss and Dice Loss for optimal contact prediction"""
    def __init__(self, focal_alpha=0.8, focal_gamma=3.0, dice_weight=0.3, bce_weight=0.7):
        super().__init__()
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.dice_loss = AdvancedDiceLoss(smooth=1.0, gamma=2.0)
        
    def focal_loss(self, inputs, targets):
        if inputs.min() >= 0 and inputs.max() <= 1:
            eps = 1e-8
            inputs = torch.log(inputs / (1 - inputs + eps))
        
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        
        alpha_weight = self.focal_alpha * targets + (1 - self.focal_alpha) * (1 - targets)
        focal_weight = (1 - pt) ** self.focal_gamma
        
        return (alpha_weight * focal_weight * bce_loss).mean()
    
    def forward(self, inputs, targets):
        focal = self.focal_loss(inputs, targets)
        dice = self.dice_loss(inputs, targets)
        
        return self.bce_weight * focal + self.dice_weight * dice

class ContactTransformerBlock(nn.Module):
    """Transformer block specifically for contact prediction"""
    def __init__(self, embed_dim=512, num_heads=8, mlp_ratio=4.0, dropout=0.1):
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
        
    def forward(self, x):
        # Self-attention
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_out
        
        # MLP
        x = x + self.mlp(self.norm2(x))
        return x

class BreakthroughContactPredictor(nn.Module):
    """Breakthrough contact predictor with radical architectural improvements"""
    
    def __init__(self, mae_checkpoint_path=None, freeze_encoder=False, contact_threshold=8.0):
        super().__init__()
        self.contact_threshold = contact_threshold
        
        # Create MAE model
        self.mae_model = ProteinDistanceMAE(
            img_size=64,
            patch_size=4,
            embed_dim=256,
            decoder_embed_dim=128,
            depth=8,
            decoder_depth=4,
            num_heads=8,
            decoder_num_heads=8
        )
        
        # Load pre-trained weights
        if mae_checkpoint_path and os.path.exists(mae_checkpoint_path):
            try:
                print(f"Loading MAE weights from {mae_checkpoint_path}")
                checkpoint = torch.load(mae_checkpoint_path, map_location='cpu', weights_only=False)
                
                if 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                    print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
                else:
                    state_dict = checkpoint
                
                self.mae_model.load_state_dict(state_dict, strict=True)
                print("✅ Successfully loaded pre-trained MAE weights!")
                
            except Exception as e:
                print(f"❌ Error loading MAE weights: {e}")
        
        self.encoder = self.mae_model.encoder
        
        # Multi-scale feature extraction
        self.features = {}
        self.register_hooks()
        
        print("🔓 Fine-tuning encoder for optimal contact prediction")
        
        # Enhanced feature projections with more capacity
        self.feature_projections = nn.ModuleDict({
            'layer_2': nn.Sequential(
                nn.Linear(256, 256),
                nn.LayerNorm(256),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(256, 128)
            ),
            'layer_4': nn.Sequential(
                nn.Linear(256, 256),
                nn.LayerNorm(256),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(256, 128)
            ),
            'layer_6': nn.Sequential(
                nn.Linear(256, 256),
                nn.LayerNorm(256),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(256, 128)
            ),
            'layer_8': nn.Sequential(
                nn.Linear(256, 256),
                nn.LayerNorm(256),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(256, 128)
            )
        })
        
        # Contact-specific transformer layers
        self.contact_transformer = nn.Sequential(
            ContactTransformerBlock(embed_dim=512, num_heads=8, dropout=0.15),
            ContactTransformerBlock(embed_dim=512, num_heads=8, dropout=0.15),
            ContactTransformerBlock(embed_dim=512, num_heads=8, dropout=0.10)
        )
        
        # Much deeper and more sophisticated prediction head
        self.contact_head = nn.Sequential(
            nn.Linear(512, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Dropout(0.3),
            
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.25),
            
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.2),
            
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.15),
            
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(0.1),
            
            nn.Linear(64, 1)  # Output logits
        )
        
        # Advanced spatial reconstruction with skip connections
        self.spatial_conv1 = nn.Sequential(
            nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2),
            nn.BatchNorm2d(256),
            nn.GELU(),
            nn.Dropout2d(0.1)
        )
        
        self.spatial_conv2 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.Dropout2d(0.1)
        )
        
        self.spatial_refine = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, 1, kernel_size=1)
        )
        
        # Learnable fusion with temperature
        self.fusion_weights = nn.Parameter(torch.tensor([0.5, 0.5]))
        self.temperature = nn.Parameter(torch.tensor(1.0))
        
    def register_hooks(self):
        """Register hooks for multi-scale feature extraction"""
        def get_features(name):
            def hook(model, input, output):
                self.features[name] = output
            return hook
        
        hook_layers = [2, 4, 6, 7]
        for layer_idx in hook_layers:
            if layer_idx < len(self.encoder.blocks):
                self.encoder.blocks[layer_idx].register_forward_hook(
                    get_features(f'layer_{layer_idx}')
                )
    
    def forward(self, distance_maps):
        batch_size = distance_maps.shape[0]
        
        # Clear features
        self.features.clear()
        
        # Get encoder features
        encoder_output = self.encoder(distance_maps, mask_ratio=0.0)
        
        if len(encoder_output) == 4:
            encoded_features, _, _, _ = encoder_output
        elif len(encoder_output) == 3:
            encoded_features, _, _ = encoder_output
        else:
            encoded_features = encoder_output[0] if isinstance(encoder_output, (tuple, list)) else encoder_output
        
        # Multi-scale feature extraction
        multi_scale_features = []
        
        available_layers = ['layer_2', 'layer_4', 'layer_6', 'layer_7']
        projection_mapping = {
            'layer_2': 'layer_2',
            'layer_4': 'layer_4', 
            'layer_6': 'layer_6',
            'layer_7': 'layer_8'
        }
        
        for layer_name in available_layers:
            if layer_name in self.features:
                layer_features = self.features[layer_name][:, 1:]  # Remove CLS
                projection_key = projection_mapping[layer_name]
                if projection_key in self.feature_projections:
                    projected = self.feature_projections[projection_key](layer_features)
                    multi_scale_features.append(projected)
        
        # Ensure 4 feature scales
        final_features = encoded_features[:, 1:]
        while len(multi_scale_features) < 4:
            projected = self.feature_projections['layer_8'](final_features)
            multi_scale_features.append(projected)
        
        # Fuse multi-scale features
        fused_features = torch.cat(multi_scale_features, dim=-1)  # [B, num_patches, 512]
        
        # Apply contact-specific transformer
        transformed_features = self.contact_transformer(fused_features)
        
        # Method 1: Advanced spatial reconstruction
        num_patches_per_side = int(np.sqrt(final_features.shape[1]))
        spatial_features = final_features.view(
            batch_size, num_patches_per_side, num_patches_per_side, -1
        ).permute(0, 3, 1, 2)  # [B, 256, 16, 16]
        
        # Multi-stage spatial reconstruction
        spatial_out1 = self.spatial_conv1(spatial_features)  # [B, 256, 32, 32]
        spatial_out2 = self.spatial_conv2(spatial_out1)      # [B, 128, 64, 64]
        contact_map_spatial = self.spatial_refine(spatial_out2)  # [B, 1, 64, 64]
        
        # Method 2: Transformer-enhanced patch prediction
        patch_contacts = self.contact_head(transformed_features)  # [B, 256, 1]
        patch_contact_map = patch_contacts.view(
            batch_size, 1, num_patches_per_side, num_patches_per_side
        )
        patch_contact_map = F.interpolate(
            patch_contact_map, size=(64, 64), mode='bilinear', align_corners=False
        )
        
        # Temperature-scaled fusion
        weights = F.softmax(self.fusion_weights * self.temperature, dim=0)
        combined_contact_map = weights[0] * contact_map_spatial + weights[1] * patch_contact_map
        
        return combined_contact_map

class RadicalContactMapDataset(Dataset):
    """Radical dataset improvements with smarter augmentation and contact definitions"""
    
    def __init__(self, distance_maps, contact_threshold=8.0, sequence_separation=12, 
                 max_length=64, augment=True, progressive_contacts=True):
        self.distance_maps = distance_maps
        self.contact_threshold = contact_threshold
        self.sequence_separation = sequence_separation
        self.max_length = max_length
        self.denorm_factor = 20.0
        self.augment = augment
        self.progressive_contacts = progressive_contacts
        
    def __len__(self):
        return len(self.distance_maps)
    
    def __getitem__(self, idx):
        dist_map = self.distance_maps[idx].clone()
        if len(dist_map.shape) == 3:
            dist_map = dist_map.squeeze(0)
        
        # More sophisticated augmentation (80% chance)
        if self.augment and random.random() > 0.2:
            dist_map = self._radical_augmentation(dist_map)
        
        # Progressive contact definition
        if self.progressive_contacts:
            contact_map = self._create_progressive_contacts(dist_map)
        else:
            dist_map_angstrom = dist_map * self.denorm_factor
            contact_map = (dist_map_angstrom <= self.contact_threshold).float()
        
        # Apply sequence separation mask
        seq_len = min(dist_map.shape[0], self.max_length)
        separation_mask = torch.zeros_like(contact_map)
        
        for i in range(seq_len):
            for j in range(seq_len):
                if abs(i - j) >= self.sequence_separation:
                    separation_mask[i, j] = 1
        
        contact_map = contact_map * separation_mask
        
        # Minimal, realistic noise
        noisy_dist_map = self._add_smart_noise(dist_map)
        
        return {
            'distance_map': noisy_dist_map.unsqueeze(0),
            'contact_map': contact_map.unsqueeze(0),
            'sequence_length': seq_len
        }
    
    def _radical_augmentation(self, dist_map):
        """Radical augmentation that preserves contact patterns"""
        # Standard geometric transforms
        if random.random() > 0.5:
            k = random.randint(0, 3)
            if k > 0:
                dist_map = torch.rot90(dist_map, k, dims=[0, 1])
        
        if random.random() > 0.4:
            dist_map = torch.flip(dist_map, dims=[1])
        
        if random.random() > 0.4:
            dist_map = dist_map.T
        
        # Contact-preserving elastic deformation
        if random.random() > 0.6:
            dist_map = self._contact_preserving_deformation(dist_map)
        
        # Local contact enhancement (make strong contacts stronger)
        if random.random() > 0.7:
            dist_map = self._enhance_local_contacts(dist_map)
            
        return dist_map
    
    def _contact_preserving_deformation(self, dist_map):
        """Deformation that preserves contact structure"""
        # Small gaussian noise with contact-aware scaling
        dist_angstrom = dist_map * self.denorm_factor
        
        # Less noise for contacts (< 8Å), more for non-contacts
        contact_mask = (dist_angstrom <= 8.0).float()
        noise_scale = 0.01 * (1 - contact_mask) + 0.005 * contact_mask
        
        noise = torch.randn_like(dist_map) * noise_scale
        return torch.clamp(dist_map + noise, 0, 1)
    
    def _enhance_local_contacts(self, dist_map):
        """Enhance local contact patterns"""
        dist_angstrom = dist_map * self.denorm_factor
        
        # Identify strong contacts (< 6Å)
        strong_contacts = (dist_angstrom <= 6.0).float()
        
        # Slightly strengthen strong contacts and weaken very distant pairs
        enhancement = strong_contacts * 0.02  # Make contacts slightly stronger
        weakening = ((dist_angstrom > 15.0).float()) * 0.01  # Make distant pairs slightly weaker
        
        enhanced_map = dist_map - enhancement + weakening
        return torch.clamp(enhanced_map, 0, 1)
    
    def _create_progressive_contacts(self, dist_map):
        """Create contact labels that improve training dynamics"""
        dist_map_angstrom = dist_map * self.denorm_factor
        
        # Multiple contact categories with smart labeling
        very_strong = (dist_map_angstrom <= 6.0).float()    # Definite contacts
        strong = ((dist_map_angstrom > 6.0) & (dist_map_angstrom <= 8.0)).float()
        medium = ((dist_map_angstrom > 8.0) & (dist_map_angstrom <= 10.0)).float()
        weak = ((dist_map_angstrom > 10.0) & (dist_map_angstrom <= 12.0)).float()
        
        # Progressive labeling: strong=1, medium=partial, weak=minimal
        contact_map = (very_strong * 1.0 + 
                      strong * 1.0 + 
                      medium * 0.6 + 
                      weak * 0.2)
        
        # Binarize for AUC compatibility: anything >= 0.8 is a contact
        contact_map = (contact_map >= 0.8).float()
        
        return contact_map
    
    def _add_smart_noise(self, dist_map):
        """Smart noise that doesn't destroy contact information"""
        # Very minimal noise for better training
        noise_level = 0.008  # Even smaller than before
        noise = torch.randn_like(dist_map) * noise_level
        
        # Apply noise more to non-contact regions
        dist_angstrom = dist_map * self.denorm_factor
        contact_regions = (dist_angstrom <= 10.0).float()
        noise = noise * (1 - 0.5 * contact_regions)  # Half noise in contact regions
        
        noisy_map = dist_map + noise
        return torch.clamp(noisy_map, 0, 1)

def train_breakthrough_contact_predictor(model, train_loader, val_loader, device, num_epochs=80, lr=1e-4):
    """Breakthrough training with optimal configuration"""
    
    model = model.to(device)
    
    # Combined loss for better contact detection
    criterion = CombinedLoss(
        focal_alpha=0.8, focal_gamma=3.0, 
        dice_weight=0.4, bce_weight=0.6
    )
    
    # Optimized training settings
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=lr, 
        weight_decay=1e-5,  # Very low weight decay
        betas=(0.9, 0.999),
        eps=1e-8
    )
    
    # Advanced learning rate schedule
    def lr_lambda(epoch):
        if epoch < 8:  # Longer warmup
            return epoch / 8
        elif epoch < 40:  # Stable high LR
            return 1.0
        else:  # Gradual decay
            return 0.5 * (1 + math.cos(math.pi * (epoch - 40) / (num_epochs - 40)))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    best_auc = 0
    patience = 20  # More patience
    patience_counter = 0
    train_history = {'train_loss': [], 'val_loss': [], 'val_auc': [], 'val_aupr': []}
    
    print(f"🚀 Starting BREAKTHROUGH training with:")
    print(f"   • Combined Loss (Focal + Dice)")
    print(f"   • Contact Transformer layers")
    print(f"   • Progressive contact labeling")
    print(f"   • Contact-preserving augmentation")
    print(f"   • Deeper architecture (1024→512→256→128→64→1)")
    print(f"   • Learning rate: {lr}")
    print(f"   • Max epochs: {num_epochs}")
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} Training"):
            distance_maps = batch['distance_map'].to(device)
            contact_maps = batch['contact_map'].to(device)
            
            optimizer.zero_grad()
            
            pred_contacts = model(distance_maps)
            loss = criterion(pred_contacts, contact_maps)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.3)  # Aggressive clipping
            optimizer.step()
            
            train_loss += loss.item()
        
        scheduler.step()
        
        # Validation
        model.eval()
        val_loss = 0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} Validation"):
                distance_maps = batch['distance_map'].to(device)
                contact_maps = batch['contact_map'].to(device)
                
                pred_contacts = model(distance_maps)
                loss = criterion(pred_contacts, contact_maps)
                
                val_loss += loss.item()
                
                pred_probs = torch.sigmoid(pred_contacts)
                all_preds.append(pred_probs.cpu().numpy())
                all_targets.append(contact_maps.cpu().numpy())
        
        # Calculate metrics
        all_preds = np.concatenate(all_preds).flatten()
        all_targets = np.concatenate(all_targets).flatten()
        
        if len(np.unique(all_targets)) > 1:
            auc = roc_auc_score(all_targets, all_preds)
            aupr = average_precision_score(all_targets, all_preds)
        else:
            auc = 0.5
            aupr = np.mean(all_targets)
        
        # Save best model
        if auc > best_auc:
            best_auc = auc
            patience_counter = 0
            os.makedirs('./contact_map_results', exist_ok=True)
            torch.save(model.state_dict(), './contact_map_results/breakthrough_contact_model.pth')
            print(f"💾 Saved BREAKTHROUGH model (AUC: {auc:.4f})")
        else:
            patience_counter += 1
        
        # Log metrics
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        train_history['train_loss'].append(avg_train_loss)
        train_history['val_loss'].append(avg_val_loss)
        train_history['val_auc'].append(auc)
        train_history['val_aupr'].append(aupr)
        
        print(f"Epoch {epoch+1}/{num_epochs}: "
              f"Train Loss: {avg_train_loss:.4f}, "
              f"Val Loss: {avg_val_loss:.4f}, "
              f"Val AUC: {auc:.4f}, "
              f"Val AUPR: {aupr:.4f}, "
              f"LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Early stopping
        if patience_counter >= patience:
            print(f"Early stopping after {patience} epochs without improvement")
            break
        
        # Stop if we reach very high performance
        if auc > 0.85:
            print(f"🎯 Achieved excellent performance (AUC > 85%)! Stopping training.")
            break
    
    return best_auc, train_history

def load_distance_maps_from_files(data_dir, max_samples=4000):
    """Load distance maps from .npz files"""
    npz_files = glob.glob(os.path.join(data_dir, "*.npz"))
    
    if len(npz_files) == 0:
        print(f"No .npz files found in {data_dir}")
        return []
    
    print(f"Found {len(npz_files)} .npz files")
    
    distance_maps = []
    for npz_file in tqdm(npz_files[:max_samples], desc="Loading distance maps"):
        try:
            data = np.load(npz_file)
            
            if len(distance_maps) == 0:
                print(f"Available keys in {os.path.basename(npz_file)}: {list(data.keys())}")
            
            dm = None
            possible_keys = ['distance_map', 'dist_map', 'distance_matrix', 'distances', 'map', 'arr_0']
            
            for key in possible_keys:
                if key in data:
                    dm = data[key]
                    if len(distance_maps) == 0:
                        print(f"Using key '{key}' for distance maps")
                    break
            
            if dm is None:
                key = list(data.keys())[0]
                dm = data[key]
                if len(distance_maps) == 0:
                    print(f"Using first available key '{key}' for distance maps")
            
            if isinstance(dm, np.ndarray):
                dm = torch.tensor(dm, dtype=torch.float32)
            
            if len(distance_maps) == 0:
                print(f"Distance map shape: {dm.shape}")
                print(f"Value range: {dm.min().item():.3f} to {dm.max().item():.3f}")
            
            if len(dm.shape) == 2:
                dm = dm.unsqueeze(0)
            elif len(dm.shape) == 3 and dm.shape[0] != 1:
                dm = dm[0:1, :, :]
            
            if dm.max() > 2.0:
                if len(distance_maps) == 0:
                    print(f"Normalizing distance map (max value: {dm.max().item():.2f})")
                dm = dm / 20.0
            
            dm = torch.clamp(dm, 0, 1)
            distance_maps.append(dm)
            
        except Exception as e:
            print(f"Error loading {npz_file}: {e}")
            continue
        
        if 'data' in locals():
            data.close()
    
    print(f"Successfully loaded {len(distance_maps)} distance maps")
    return distance_maps

def main():
    """Main function with breakthrough configuration"""
    
    data_dir = "./distance_maps"
    results_dir = "./contact_map_results"
    mae_model_path = "../best_model.pth"
    
    if not os.path.exists(mae_model_path):
        mae_model_path = "./mae_results/best_model.pth"
        if not os.path.exists(mae_model_path):
            print(f"⚠️  MAE model not found. Using random initialization.")
            mae_model_path = None
    
    os.makedirs(results_dir, exist_ok=True)
    
    # Load even MORE data for better training
    print("Loading distance maps...")
    distance_maps = load_distance_maps_from_files(data_dir, max_samples=4000)
    
    if len(distance_maps) == 0:
        print("❌ No distance maps loaded!")
        return
    
    # Create breakthrough dataset
    print("Creating BREAKTHROUGH contact map dataset...")
    dataset = RadicalContactMapDataset(
        distance_maps, 
        contact_threshold=8.0,
        sequence_separation=12,
        augment=True,
        progressive_contacts=True
    )
    
    # Optimal data split for more training data
    train_size = int(0.8 * len(dataset))  # Even more training data
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    print(f"Dataset split: {len(train_dataset)} train, {len(val_dataset)} val, {len(test_dataset)} test")
    
    # Optimized data loaders
    train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=10, shuffle=False, num_workers=0, pin_memory=True)
    
    # Initialize BREAKTHROUGH model
    print("Initializing BREAKTHROUGH contact map predictor...")
    model = BreakthroughContactPredictor(mae_model_path, freeze_encoder=False)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Train BREAKTHROUGH model
    print("Training BREAKTHROUGH contact map predictor...")
    best_auc, history = train_breakthrough_contact_predictor(
        model, train_loader, val_loader, device, num_epochs=80, lr=1e-4
    )
    
    print(f"\n🎉 BREAKTHROUGH training completed!")
    print(f"🏆 Best validation AUC: {best_auc:.4f}")
    
    # Analysis
    print(f"\n📊 BREAKTHROUGH Result Analysis:")
    if best_auc > 0.85:
        print("🌟🌟🌟 AUC 85%+ - EXCEPTIONAL performance! Publication-ready!")
    elif best_auc > 0.78:
        print("🌟🌟 AUC 78-85% - OUTSTANDING performance!")
    elif best_auc > 0.75:
        print("🌟 AUC 75-78% - EXCELLENT performance!")
    elif best_auc > 0.72:
        print("✅ AUC 72-75% - VERY GOOD performance!")
    else:
        print("⚠️  AUC < 72% - Consider ensemble methods or more data")
    
    baseline_auc = 0.6595
    improvement = best_auc - baseline_auc
    print(f"📈 Total improvement: +{improvement:.4f} AUC ({improvement*100:.1f}%)")
    
    current_best = 0.6957  # Your current best from previous run
    further_improvement = best_auc - current_best
    print(f"🚀 Improvement over current best: +{further_improvement:.4f} AUC ({further_improvement*100:.1f}%)")

if __name__ == "__main__":
    main()
