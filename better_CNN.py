import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os
import time
import numpy as np
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim
import json
import logging
from datetime import datetime
from protein_fragment_class import get_dataloaders

# Set up logging
def setup_logging(output_dir):
    """Setup comprehensive logging"""
    log_dir = os.path.join(output_dir, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f'training_{timestamp}.log')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

class ProteinDistanceMapCNN(nn.Module):
    """Optimized CNN for protein distance map reconstruction with protein-specific features"""
    
    def __init__(self, base_channels=32):
        super(ProteinDistanceMapCNN, self).__init__()
        
        # Symmetric convolution block for distance maps
        def conv_block(in_ch, out_ch, kernel_size=3, stride=1, padding=1):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, kernel_size, 1, padding, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True)
            )
        
        # Encoder with skip connections and progressive downsampling
        self.encoder1 = conv_block(1, base_channels)  # 64x64
        self.pool1 = nn.MaxPool2d(2, 2)
        
        self.encoder2 = conv_block(base_channels, base_channels*2)  # 32x32
        self.pool2 = nn.MaxPool2d(2, 2)
        
        self.encoder3 = conv_block(base_channels*2, base_channels*4)  # 16x16
        self.pool3 = nn.MaxPool2d(2, 2)
        
        self.encoder4 = conv_block(base_channels*4, base_channels*8)  # 8x8
        self.pool4 = nn.MaxPool2d(2, 2)
        
        # Bottleneck with attention mechanism for long-range dependencies
        self.bottleneck = conv_block(base_channels*8, base_channels*16)  # 4x4
        
        # Attention module (separate from bottleneck)
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(base_channels*16, base_channels*8, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels*8, base_channels*16, 1),
            nn.Sigmoid()
        )
        
        # Decoder with skip connections
        self.upconv4 = nn.ConvTranspose2d(base_channels*16, base_channels*8, 2, 2)
        self.decoder4 = conv_block(base_channels*16, base_channels*8)
        
        self.upconv3 = nn.ConvTranspose2d(base_channels*8, base_channels*4, 2, 2)
        self.decoder3 = conv_block(base_channels*8, base_channels*4)
        
        self.upconv2 = nn.ConvTranspose2d(base_channels*4, base_channels*2, 2, 2)
        self.decoder2 = conv_block(base_channels*4, base_channels*2)
        
        self.upconv1 = nn.ConvTranspose2d(base_channels*2, base_channels, 2, 2)
        self.decoder1 = conv_block(base_channels*2, base_channels)
        
        # Final output layer with symmetric constraint
        self.final_conv = nn.Sequential(
            nn.Conv2d(base_channels, base_channels//2, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels//2, 1, 1),
            nn.Sigmoid()
        )
        
        # Dropout for regularization
        self.dropout = nn.Dropout2d(0.1)
        
    def forward(self, x):
        # Encoder path with skip connections
        enc1 = self.encoder1(x)
        enc1_pool = self.pool1(enc1)
        
        enc2 = self.encoder2(enc1_pool)
        enc2_pool = self.pool2(enc2)
        
        enc3 = self.encoder3(enc2_pool)
        enc3_pool = self.pool3(enc3)
        
        enc4 = self.encoder4(enc3_pool)
        enc4_pool = self.pool4(enc4)
        
        # Bottleneck with attention
        bottleneck = self.bottleneck(enc4_pool)
        
        # Apply attention
        attention_weights = self.attention(bottleneck)
        bottleneck = bottleneck * attention_weights
        
        # Decoder path with skip connections
        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat([dec4, enc4], dim=1)
        dec4 = self.decoder4(dec4)
        dec4 = self.dropout(dec4)
        
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat([dec3, enc3], dim=1)
        dec3 = self.decoder3(dec3)
        dec3 = self.dropout(dec3)
        
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat([dec2, enc2], dim=1)
        dec2 = self.decoder2(dec2)
        
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat([dec1, enc1], dim=1)
        dec1 = self.decoder1(dec1)
        
        # Final output
        output = self.final_conv(dec1)
        
        # Enforce symmetry for distance maps
        output = self.enforce_symmetry(output)
        
        return output
    
    def enforce_symmetry(self, x):
        """Enforce symmetry constraint for distance maps"""
        return (x + x.transpose(-1, -2)) / 2

class ProteinSpecificLoss(nn.Module):
    """Custom loss function for protein distance maps"""
    
    def __init__(self, mse_weight=1.0, ssim_weight=0.3, symmetry_weight=0.1):
        super().__init__()
        self.mse_weight = mse_weight
        self.ssim_weight = ssim_weight
        self.symmetry_weight = symmetry_weight
        self.mse_loss = nn.MSELoss()
        
    def forward(self, pred, target):
        # MSE loss
        mse = self.mse_loss(pred, target)
        
        # SSIM loss (batch-wise)
        ssim_loss = 0
        for i in range(pred.size(0)):
            pred_np = pred[i, 0].detach().cpu().numpy()
            target_np = target[i, 0].detach().cpu().numpy()
            ssim_val = ssim(target_np, pred_np, data_range=1.0)
            ssim_loss += (1 - ssim_val)
        ssim_loss /= pred.size(0)
        
        # Symmetry loss
        pred_T = pred.transpose(-1, -2)
        symmetry_loss = self.mse_loss(pred, pred_T)
        
        total_loss = (self.mse_weight * mse + 
                     self.ssim_weight * ssim_loss + 
                     self.symmetry_weight * symmetry_loss)
        
        return total_loss, mse, ssim_loss, symmetry_loss

def create_protein_mask(distance_map, mask_ratio=0.75, mask_type='structured'):
    """Create protein-aware masks"""
    batch_size, channels, height, width = distance_map.shape
    mask = torch.ones_like(distance_map)
    
    for b in range(batch_size):
        if mask_type == 'structured':
            # Create structured masks that respect protein geometry
            num_blocks = int(np.sqrt(mask_ratio * height * width / 64))
            for _ in range(num_blocks):
                block_size = np.random.randint(4, 12)
                start_i = np.random.randint(0, height - block_size)
                start_j = np.random.randint(0, width - block_size)
                mask[b, :, start_i:start_i+block_size, start_j:start_j+block_size] = 0
        
        elif mask_type == 'diagonal_aware':
            # Mask while preserving some diagonal information
            flat_mask = torch.rand(height * width) > mask_ratio
            flat_mask = flat_mask.view(height, width)
            # Preserve diagonal with some probability
            diagonal_preserve = torch.rand(min(height, width)) > 0.5
            for i in range(min(height, width)):
                if diagonal_preserve[i]:
                    flat_mask[i, i] = True
            mask[b, 0] = flat_mask.float()
        
        else:  # random
            flat_mask = torch.rand(height * width) > mask_ratio
            mask[b, 0] = flat_mask.view(height, width).float()
    
    return mask

def train_optimized_protein_cnn(data_dir, output_dir, epochs=50, batch_size=32, 
                               learning_rate=0.001, mask_ratio=0.75):
    """Train optimized protein distance map CNN"""
    
    # Setup
    os.makedirs(output_dir, exist_ok=True)
    logger = setup_logging(output_dir)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Define split files
    split_files = {
        'train': './splits/train.txt',
        'val': './splits/val.txt',
        'test': './splits/test.txt'
    }
    
    # Get dataloaders
    logger.info("Loading datasets...")
    dataloaders = get_dataloaders(data_dir, batch_size=batch_size, split_files=split_files)
    
    # Create model
    model = ProteinDistanceMapCNN(base_channels=32).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model created with {total_params} total parameters ({trainable_params} trainable)")
    
    # Loss function and optimizer with scheduling
    criterion = ProteinSpecificLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, 
                                                    factor=0.5, verbose=True)
    
    # Training metrics tracking
    metrics = {
        'train_losses': [], 'val_losses': [], 'train_mse': [], 'val_mse': [],
        'train_ssim': [], 'val_ssim': [], 'train_symmetry': [], 'val_symmetry': [],
        'learning_rates': [], 'epoch_times': []
    }
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    logger.info(f"Starting training for {epochs} epochs with mask ratio {mask_ratio}")
    
    for epoch in range(epochs):
        epoch_start_time = time.time()
        
        # Progressive mask ratio (start easier, get harder)
        current_mask_ratio = min(mask_ratio, 0.5 + (mask_ratio - 0.5) * epoch / (epochs * 0.3))
        
        # Training phase
        model.train()
        train_metrics = {'total_loss': 0, 'mse': 0, 'ssim': 0, 'symmetry': 0, 'count': 0}
        
        for batch in tqdm(dataloaders['train'], desc=f"Epoch {epoch+1}/{epochs} (Train)"):
            distance_maps = batch['distance_map'].to(device)
            
            # Create protein-aware masks
            mask = create_protein_mask(distance_maps, current_mask_ratio, 'structured')
            masked_input = distance_maps * mask.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            reconstructions = model(masked_input)
            total_loss, mse_loss, ssim_loss, symmetry_loss = criterion(reconstructions, distance_maps)
            
            # Backward pass
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # Update metrics
            train_metrics['total_loss'] += total_loss.item()
            train_metrics['mse'] += mse_loss.item()
            train_metrics['ssim'] += ssim_loss
            train_metrics['symmetry'] += symmetry_loss.item()
            train_metrics['count'] += 1
        
        # Average training metrics
        for key in train_metrics:
            if key != 'count':
                train_metrics[key] /= train_metrics['count']
        
        # Validation phase
        model.eval()
        val_metrics = {'total_loss': 0, 'mse': 0, 'ssim': 0, 'symmetry': 0, 'count': 0}
        
        with torch.no_grad():
            for batch in tqdm(dataloaders['val'], desc=f"Epoch {epoch+1}/{epochs} (Val)"):
                distance_maps = batch['distance_map'].to(device)
                
                # Create masks for validation
                mask = create_protein_mask(distance_maps, mask_ratio, 'structured')
                masked_input = distance_maps * mask.to(device)
                
                reconstructions = model(masked_input)
                total_loss, mse_loss, ssim_loss, symmetry_loss = criterion(reconstructions, distance_maps)
                
                val_metrics['total_loss'] += total_loss.item()
                val_metrics['mse'] += mse_loss.item()
                val_metrics['ssim'] += ssim_loss
                val_metrics['symmetry'] += symmetry_loss.item()
                val_metrics['count'] += 1
        
        # Average validation metrics
        for key in val_metrics:
            if key != 'count':
                val_metrics[key] /= val_metrics['count']
        
        # Update learning rate
        scheduler.step(val_metrics['total_loss'])
        current_lr = optimizer.param_groups[0]['lr']
        
        # Record metrics
        epoch_time = time.time() - epoch_start_time
        metrics['train_losses'].append(train_metrics['total_loss'])
        metrics['val_losses'].append(val_metrics['total_loss'])
        metrics['train_mse'].append(train_metrics['mse'])
        metrics['val_mse'].append(val_metrics['mse'])
        metrics['train_ssim'].append(train_metrics['ssim'])
        metrics['val_ssim'].append(val_metrics['ssim'])
        metrics['train_symmetry'].append(train_metrics['symmetry'])
        metrics['val_symmetry'].append(val_metrics['symmetry'])
        metrics['learning_rates'].append(current_lr)
        metrics['epoch_times'].append(epoch_time)
        
        # Log progress
        logger.info(f"Epoch {epoch+1}/{epochs} - "
                   f"Train Loss: {train_metrics['total_loss']:.6f}, "
                   f"Val Loss: {val_metrics['total_loss']:.6f}, "
                   f"Train MSE: {train_metrics['mse']:.6f}, "
                   f"Val MSE: {val_metrics['mse']:.6f}, "
                   f"Val SSIM Loss: {val_metrics['ssim']:.6f}, "
                   f"LR: {current_lr:.6f}, "
                   f"Time: {epoch_time:.1f}s, "
                   f"Mask Ratio: {current_mask_ratio:.3f}")
        
        # Save best model
        if val_metrics['total_loss'] < best_val_loss:
            best_val_loss = val_metrics['total_loss']
            patience_counter = 0
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_loss': best_val_loss,
                'metrics': metrics
            }, os.path.join(output_dir, 'best_model.pth'))
            logger.info(f"New best model saved with validation loss: {best_val_loss:.6f}")
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= 10:
            logger.info(f"Early stopping triggered after {epoch+1} epochs")
            break
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            checkpoint_path = os.path.join(output_dir, f'checkpoint_epoch_{epoch+1}.pth')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'metrics': metrics
            }, checkpoint_path)
    
    # Save final model and metrics
    torch.save(model.state_dict(), os.path.join(output_dir, 'final_model.pth'))
    
    # Save training metrics
    with open(os.path.join(output_dir, 'training_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Create comprehensive plots
    create_training_plots(metrics, output_dir)
    
    # Evaluate on test set
    logger.info("Evaluating on test set...")
    test_metrics = evaluate_comprehensive(model, dataloaders['test'], device, criterion, 
                                        output_dir, mask_ratio)
    
    # Save test results
    with open(os.path.join(output_dir, 'test_results.json'), 'w') as f:
        json.dump(test_metrics, f, indent=2)
    
    logger.info("Training completed!")
    logger.info(f"Best validation loss: {best_val_loss:.6f}")
    logger.info(f"Test MSE: {test_metrics['mse']:.6f}")
    logger.info(f"Test SSIM: {test_metrics['ssim']:.6f}")
    logger.info(f"Average reconstruction time: {test_metrics['avg_time_ms']:.2f} ms")
    
    return model, metrics, test_metrics

def evaluate_comprehensive(model, dataloader, device, criterion, output_dir, mask_ratio):
    """Comprehensive evaluation with detailed metrics and visualizations"""
    model.eval()
    
    all_metrics = {
        'total_losses': [], 'mse_losses': [], 'ssim_losses': [], 'symmetry_losses': [],
        'reconstruction_times': [], 'ssim_scores': []
    }
    
    # For visualization samples
    sample_originals = []
    sample_reconstructions = []
    sample_masks = []
    sample_errors = []
    
    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader, desc="Evaluating")):
            distance_maps = batch['distance_map'].to(device)
            
            # Create masks
            mask = create_protein_mask(distance_maps, mask_ratio, 'structured')
            masked_input = distance_maps * mask.to(device)
            
            # Time reconstruction
            start_time = time.time()
            reconstructions = model(masked_input)
            end_time = time.time()
            
            reconstruction_time = (end_time - start_time) * 1000 / distance_maps.size(0)
            all_metrics['reconstruction_times'].append(reconstruction_time)
            
            # Calculate losses
            total_loss, mse_loss, ssim_loss, symmetry_loss = criterion(reconstructions, distance_maps)
            all_metrics['total_losses'].append(total_loss.item())
            all_metrics['mse_losses'].append(mse_loss.item())
            all_metrics['ssim_losses'].append(ssim_loss)
            all_metrics['symmetry_losses'].append(symmetry_loss.item())
            
            # Calculate SSIM scores (higher is better)
            for j in range(distance_maps.size(0)):
                orig = distance_maps[j, 0].cpu().numpy()
                recon = reconstructions[j, 0].cpu().numpy()
                ssim_score = ssim(orig, recon, data_range=1.0)
                all_metrics['ssim_scores'].append(ssim_score)
            
            # Collect samples for visualization (first 10 batches)
            if i < 10:
                sample_originals.extend(distance_maps[:3].cpu())
                sample_reconstructions.extend(reconstructions[:3].cpu())
                sample_masks.extend(mask[:3].cpu())
                
                # Calculate error maps
                for j in range(min(3, distance_maps.size(0))):
                    error = torch.abs(distance_maps[j] - reconstructions[j]).cpu()
                    sample_errors.append(error)
    
    # Calculate final metrics
    final_metrics = {
        'total_loss': np.mean(all_metrics['total_losses']),
        'mse': np.mean(all_metrics['mse_losses']),
        'ssim_loss': np.mean(all_metrics['ssim_losses']),
        'symmetry_loss': np.mean(all_metrics['symmetry_losses']),
        'ssim': np.mean(all_metrics['ssim_scores']),
        'avg_time_ms': np.mean(all_metrics['reconstruction_times']),
        'std_time_ms': np.std(all_metrics['reconstruction_times'])
    }
    
    # Create comprehensive visualizations
    create_evaluation_visualizations(sample_originals, sample_reconstructions, 
                                   sample_masks, sample_errors, output_dir)
    
    # Create metric distribution plots
    create_metric_distributions(all_metrics, output_dir)
    
    return final_metrics

def create_training_plots(metrics, output_dir):
    """Create comprehensive training plots"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Loss curves
    axes[0, 0].plot(metrics['train_losses'], label='Train Total Loss', alpha=0.8)
    axes[0, 0].plot(metrics['val_losses'], label='Val Total Loss', alpha=0.8)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Total Loss')
    axes[0, 0].set_title('Training/Validation Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # MSE curves
    axes[0, 1].plot(metrics['train_mse'], label='Train MSE', alpha=0.8)
    axes[0, 1].plot(metrics['val_mse'], label='Val MSE', alpha=0.8)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('MSE Loss')
    axes[0, 1].set_title('MSE Loss Curves')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # SSIM curves
    axes[0, 2].plot(metrics['train_ssim'], label='Train SSIM Loss', alpha=0.8)
    axes[0, 2].plot(metrics['val_ssim'], label='Val SSIM Loss', alpha=0.8)
    axes[0, 2].set_xlabel('Epoch')
    axes[0, 2].set_ylabel('SSIM Loss')
    axes[0, 2].set_title('SSIM Loss Curves')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # Symmetry loss
    axes[1, 0].plot(metrics['train_symmetry'], label='Train Symmetry', alpha=0.8)
    axes[1, 0].plot(metrics['val_symmetry'], label='Val Symmetry', alpha=0.8)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Symmetry Loss')
    axes[1, 0].set_title('Symmetry Loss Curves')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Learning rate
    axes[1, 1].plot(metrics['learning_rates'], alpha=0.8)
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Learning Rate')
    axes[1, 1].set_title('Learning Rate Schedule')
    axes[1, 1].set_yscale('log')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Epoch times
    axes[1, 2].plot(metrics['epoch_times'], alpha=0.8)
    axes[1, 2].set_xlabel('Epoch')
    axes[1, 2].set_ylabel('Time (seconds)')
    axes[1, 2].set_title('Training Time per Epoch')
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_curves.png'), dpi=300, bbox_inches='tight')
    plt.close()

def create_evaluation_visualizations(originals, reconstructions, masks, errors, output_dir):
    """Create comprehensive evaluation visualizations"""
    num_samples = min(12, len(originals))
    
    fig, axes = plt.subplots(num_samples, 4, figsize=(16, 4*num_samples))
    
    for i in range(num_samples):
        # Original
        im1 = axes[i, 0].imshow(originals[i][0], cmap='viridis', vmin=0, vmax=1)
        axes[i, 0].set_title('Original Distance Map')
        axes[i, 0].axis('off')
        
        # Masked input
        masked = originals[i][0] * masks[i][0]
        im2 = axes[i, 1].imshow(masked, cmap='viridis', vmin=0, vmax=1)
        axes[i, 1].set_title('Masked Input (75%)')
        axes[i, 1].axis('off')
        
        # Reconstruction
        im3 = axes[i, 2].imshow(reconstructions[i][0], cmap='viridis', vmin=0, vmax=1)
        axes[i, 2].set_title('Reconstruction')
        axes[i, 2].axis('off')
        
        # Error map
        error_map = errors[i][0]
        im4 = axes[i, 3].imshow(error_map, cmap='hot', vmin=0)
        axes[i, 3].set_title(f'Error (Mean: {error_map.mean():.4f})')
        axes[i, 3].axis('off')
        
        # Add colorbars
        if i == 0:
            plt.colorbar(im1, ax=axes[i, 0], fraction=0.046, pad=0.04)
            plt.colorbar(im2, ax=axes[i, 1], fraction=0.046, pad=0.04)
            plt.colorbar(im3, ax=axes[i, 2], fraction=0.046, pad=0.04)
            plt.colorbar(im4, ax=axes[i, 3], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'reconstruction_examples.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()

def create_metric_distributions(metrics, output_dir):
    """Create distribution plots for evaluation metrics"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # SSIM distribution
    axes[0, 0].hist(metrics['ssim_scores'], bins=50, alpha=0.7, edgecolor='black')
    axes[0, 0].set_xlabel('SSIM Score')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title(f'SSIM Distribution (Mean: {np.mean(metrics["ssim_scores"]):.4f})')
    axes[0, 0].grid(True, alpha=0.3)
    
    # MSE distribution
    axes[0, 1].hist(metrics['mse_losses'], bins=50, alpha=0.7, edgecolor='black')
    axes[0, 1].set_xlabel('MSE Loss')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title(f'MSE Distribution (Mean: {np.mean(metrics["mse_losses"]):.6f})')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Reconstruction time distribution
    axes[1, 0].hist(metrics['reconstruction_times'], bins=50, alpha=0.7, edgecolor='black')
    axes[1, 0].set_xlabel('Reconstruction Time (ms)')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title(f'Reconstruction Time Distribution (Mean: {np.mean(metrics["reconstruction_times"]):.2f}ms)')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Total loss distribution
    axes[1, 1].hist(metrics['total_losses'], bins=50, alpha=0.7, edgecolor='black')
    axes[1, 1].set_xlabel('Total Loss')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title(f'Total Loss Distribution (Mean: {np.mean(metrics["total_losses"]):.6f})')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'metric_distributions.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()

def analyze_distance_map_properties(originals, reconstructions, output_dir):
    """Analyze specific properties of protein distance maps"""
    
    diagonal_correlations = []
    off_diagonal_correlations = []
    
    for orig, recon in zip(originals[:100], reconstructions[:100]):  # Analyze first 100
        orig_np = orig[0].numpy()
        recon_np = recon[0].numpy()
        
        # Diagonal vs off-diagonal correlation
        mask = np.triu(np.ones_like(orig_np, dtype=bool), k=1)  # Upper triangle
        diag_mask = np.eye(orig_np.shape[0], dtype=bool)
        
        # Correlation along diagonal
        if np.any(diag_mask):
            diag_corr = np.corrcoef(orig_np[diag_mask], recon_np[diag_mask])[0, 1]
            if not np.isnan(diag_corr):
                diagonal_correlations.append(diag_corr)
        
        # Correlation off-diagonal
        if np.any(mask):
            off_diag_corr = np.corrcoef(orig_np[mask], recon_np[mask])[0, 1]
            if not np.isnan(off_diag_corr):
                off_diagonal_correlations.append(off_diag_corr)
    
    # Create analysis plots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Diagonal correlations
    axes[0].hist(diagonal_correlations, bins=30, alpha=0.7, edgecolor='black')
    axes[0].set_xlabel('Correlation Coefficient')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title(f'Diagonal Correlations (Mean: {np.mean(diagonal_correlations):.4f})')
    axes[0].grid(True, alpha=0.3)
    
    # Off-diagonal correlations
    axes[1].hist(off_diagonal_correlations, bins=30, alpha=0.7, edgecolor='black')
    axes[1].set_xlabel('Correlation Coefficient')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title(f'Off-Diagonal Correlations (Mean: {np.mean(off_diagonal_correlations):.4f})')
    axes[1].grid(True, alpha=0.3)
    
    # Comparison plot
    axes[2].boxplot([diagonal_correlations, off_diagonal_correlations], 
                      labels=['Diagonal', 'Off-Diagonal'])
    axes[2].set_ylabel('Correlation Coefficient')
    axes[2].set_title('Diagonal vs Off-Diagonal Correlation')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'distance_map_analysis.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # Return analysis results
    analysis_results = {
        'diagonal_correlation_mean': np.mean(diagonal_correlations),
        'off_diagonal_correlation_mean': np.mean(off_diagonal_correlations),
        'diagonal_correlation_std': np.std(diagonal_correlations),
        'off_diagonal_correlation_std': np.std(off_diagonal_correlations)
    }
    
    return analysis_results

def create_model_comparison_plot(output_dir):
    """Create a comparison plot with the baseline results"""
    
    # Baseline results from the paper
    baseline_results = {
        'Standard CNN': {'MSE': 0.000214, 'SSIM': 0.983256, 'Time_ms': 2.81, 'Params_M': 0.27},
        'Masked CNN (75%)': {'MSE': 0.000575, 'SSIM': 0.959629, 'Time_ms': 0.93, 'Params_M': 0.27},
        'Vision Transformer': {'MSE': 0.125557, 'SSIM': 0.009061, 'Time_ms': 51.39, 'Params_M': 105.10}
    }
    
    # Create comparison plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    models = list(baseline_results.keys())
    mse_values = [baseline_results[m]['MSE'] for m in models]
    ssim_values = [baseline_results[m]['SSIM'] for m in models]
    time_values = [baseline_results[m]['Time_ms'] for m in models]
    param_values = [baseline_results[m]['Params_M'] for m in models]
    
    # MSE comparison
    bars1 = axes[0, 0].bar(models, mse_values, alpha=0.7)
    axes[0, 0].set_ylabel('MSE Loss')
    axes[0, 0].set_title('MSE Comparison')
    axes[0, 0].tick_params(axis='x', rotation=45)
    axes[0, 0].set_yscale('log')
    
    # SSIM comparison
    bars2 = axes[0, 1].bar(models, ssim_values, alpha=0.7)
    axes[0, 1].set_ylabel('SSIM Score')
    axes[0, 1].set_title('SSIM Comparison')
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # Time comparison
    bars3 = axes[1, 0].bar(models, time_values, alpha=0.7)
    axes[1, 0].set_ylabel('Time (ms)')
    axes[1, 0].set_title('Reconstruction Time Comparison')
    axes[1, 0].tick_params(axis='x', rotation=45)
    axes[1, 0].set_yscale('log')
    
    # Parameter comparison
    bars4 = axes[1, 1].bar(models, param_values, alpha=0.7)
    axes[1, 1].set_ylabel('Parameters (M)')
    axes[1, 1].set_title('Model Size Comparison')
    axes[1, 1].tick_params(axis='x', rotation=45)
    axes[1, 1].set_yscale('log')
    
    # Add value labels on bars
    for ax, bars, values in zip([axes[0,0], axes[0,1], axes[1,0], axes[1,1]], 
                               [bars1, bars2, bars3, bars4], 
                               [mse_values, ssim_values, time_values, param_values]):
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{value:.3f}' if value < 1 else f'{value:.1f}',
                   ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'model_comparison.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()

def generate_comprehensive_report(model, metrics, test_metrics, output_dir):
    """Generate a comprehensive training and evaluation report"""
    
    report = f"""
# Optimized Protein Distance Map CNN - Training Report

## Model Architecture
- **Base Architecture**: U-Net style encoder-decoder with skip connections
- **Key Features**: 
  - Protein-specific symmetry enforcement
  - Attention mechanism in bottleneck
  - Progressive masking during training
  - Multi-component loss function (MSE + SSIM + Symmetry)
- **Parameters**: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M trainable parameters

## Training Configuration
- **Epochs**: {len(metrics['train_losses'])}
- **Mask Ratio**: 75% (progressive from 50%)
- **Batch Size**: 32
- **Optimizer**: AdamW with weight decay
- **Scheduler**: ReduceLROnPlateau
- **Early Stopping**: Patience = 10

## Training Results
- **Best Validation Loss**: {min(metrics['val_losses']):.6f}
- **Final Training Loss**: {metrics['train_losses'][-1]:.6f}
- **Final Validation Loss**: {metrics['val_losses'][-1]:.6f}
- **Training Time**: {sum(metrics['epoch_times']):.1f} seconds total

## Test Set Performance
- **MSE Loss**: {test_metrics['mse']:.6f}
- **SSIM Score**: {test_metrics['ssim']:.6f}
- **SSIM Loss**: {test_metrics['ssim_loss']:.6f}
- **Symmetry Loss**: {test_metrics['symmetry_loss']:.6f}
- **Average Reconstruction Time**: {test_metrics['avg_time_ms']:.2f} ± {test_metrics['std_time_ms']:.2f} ms

## Key Improvements Over Baseline
1. **Skip Connections**: Better gradient flow and detail preservation
2. **Symmetry Enforcement**: Ensures distance map properties are maintained
3. **Multi-Component Loss**: Combines pixel-wise accuracy with perceptual quality
4. **Progressive Masking**: Curriculum learning approach
5. **Attention Mechanism**: Better long-range dependency modeling

## Files Generated
- `best_model.pth`: Best model checkpoint
- `final_model.pth`: Final model weights
- `training_metrics.json`: Complete training history
- `test_results.json`: Comprehensive test results
- `training_curves.png`: Training visualization
- `reconstruction_examples.png`: Sample reconstructions
- `metric_distributions.png`: Statistical analysis
- `distance_map_analysis.png`: Protein-specific analysis
- `model_comparison.png`: Comparison with baselines

## Recommendations for Further Improvement
1. Implement 3D structure evaluation via multidimensional scaling
2. Add contact map prediction as auxiliary task
3. Experiment with transformer-based attention layers
4. Implement fragment similarity matching evaluation
5. Test on larger protein fragments (60-80 residues)
"""
    
    with open(os.path.join(output_dir, 'training_report.md'), 'w') as f:
        f.write(report)

if __name__ == "__main__":
    # Configuration
    data_dir = "./distance_maps"
    output_dir = "./optimized_protein_cnn_results"
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Train the optimized model
    print("Starting optimized protein distance map CNN training...")
    model, training_metrics, test_metrics = train_optimized_protein_cnn(
        data_dir=data_dir,
        output_dir=output_dir,
        epochs=50,
        batch_size=32,
        learning_rate=0.001,
        mask_ratio=0.75
    )
    
    # Generate additional analysis
    print("Generating additional analysis...")
    
    # Load some samples for analysis
    split_files = {
        'train': './splits/train.txt',
        'val': './splits/val.txt',
        'test': './splits/test.txt'
    }
    dataloaders = get_dataloaders(data_dir, batch_size=32, split_files=split_files)
    
    # Get sample data for analysis
    sample_batch = next(iter(dataloaders['test']))
    sample_maps = sample_batch['distance_map']
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    
    with torch.no_grad():
        # Create masks and reconstruct
        mask = create_protein_mask(sample_maps, 0.75, 'structured')
        masked_input = sample_maps * mask
        reconstructions = model(masked_input.to(device))
        reconstructions = reconstructions.cpu()
    
    # Protein-specific analysis
    analysis_results = analyze_distance_map_properties(
        sample_maps, reconstructions, output_dir
    )
    
    # Save analysis results
    with open(os.path.join(output_dir, 'protein_analysis.json'), 'w') as f:
        json.dump(analysis_results, f, indent=2)
    
    # Create model comparison plot
    create_model_comparison_plot(output_dir)
    
    # Generate comprehensive report
    generate_comprehensive_report(model, training_metrics, test_metrics, output_dir)
    
    print("\n" + "="*60)
    print("OPTIMIZED PROTEIN CNN TRAINING COMPLETE!")
    print("="*60)
    print(f"Results saved to: {output_dir}")
    print(f"Test MSE: {test_metrics['mse']:.6f}")
    print(f"Test SSIM: {test_metrics['ssim']:.6f}")
    print(f"Average reconstruction time: {test_metrics['avg_time_ms']:.2f} ms")
    print("="*60)
