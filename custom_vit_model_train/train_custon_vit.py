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
import sys
sys.path.append('..')  # Add parent directory to Python path
from protein_fragment_class import get_dataloaders
# import wandb  # Optional: for experiment tracking

# Import our custom MAE model
from custom_Vit import (
    ProteinDistanceMAE, ProteinMAELoss, 
    create_optimizer, cosine_scheduler
)

def check_cuda_availability():
    """Check CUDA availability and print detailed information"""
    print("\n=== CUDA Availability Check ===")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        print("\nGPU Information:")
        for i in range(torch.cuda.device_count()):
            print(f"\nGPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"Memory allocated: {torch.cuda.memory_allocated(i) / 1024**2:.2f} MB")
            print(f"Memory reserved: {torch.cuda.memory_reserved(i) / 1024**2:.2f} MB")
        
        # Set default device
        torch.cuda.set_device(0)
        print(f"\nUsing GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("\nWARNING: CUDA is not available. Training will be very slow on CPU!")
        print("Please check your CUDA installation if you intended to use GPU.")
    
    print("==============================\n")

def print_gpu_utilization():
    """Print current GPU memory usage"""
    if torch.cuda.is_available():
        print("\nGPU Memory Usage:")
        print(f"Device: {torch.cuda.get_device_name(0)}")
        print(f"Memory Allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
        print(f"Memory Reserved: {torch.cuda.memory_reserved(0) / 1024**2:.2f} MB")
        print(f"Max Memory Allocated: {torch.cuda.max_memory_allocated(0) / 1024**2:.2f} MB")
    else:
        print("No GPU available")

def train_protein_mae(
    data_dir="./distance_maps",
    output_dir="./mae_results",
    epochs=100,
    batch_size=128,
    learning_rate=1e-4,
    mask_ratio=0.75,
    mask_ratio_schedule=None,
    warmup_epochs=10,
    use_wandb=False,
    num_workers=4,
    pin_memory=True
):
    """Train the Protein Distance MAE model"""
    
    # Check CUDA availability first
    check_cuda_availability()
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Print initial GPU stats
    print_gpu_utilization()
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize wandb if requested
    if use_wandb:
        import wandb
        wandb.init(project="protein-mae", config={
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "mask_ratio": mask_ratio,
            "architecture": "ProteinDistanceMAE"
        })
    
    # Define split files
    split_files = {
        'train': './splits/train.txt',
        'val': './splits/val.txt',
        'test': './splits/test.txt'
    }
    
    # Get dataloaders
    print("Loading datasets...")
    dataloaders = get_dataloaders(data_dir, batch_size=batch_size, split_files=split_files, num_workers=num_workers, pin_memory=pin_memory)
    
    # Create model
    model = ProteinDistanceMAE(
        img_size=64,
        patch_size=4,
        embed_dim=256,
        decoder_embed_dim=128,
        depth=8,
        decoder_depth=4,
        num_heads=8,
        decoder_num_heads=8
    ).to(device)
    
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Print GPU stats after model creation
    print("\nGPU stats after model creation:")
    print_gpu_utilization()
    
    # Loss function
    criterion = ProteinMAELoss(smoothness_weight=0.1, symmetry_weight=0.1)
    
    # Optimizer with cosine LR schedule
    optimizer = create_optimizer(model, lr=learning_rate, weight_decay=0.05)
    
    # Learning rate schedule
    num_training_steps = epochs * len(dataloaders['train'])
    lr_schedule = cosine_scheduler(
        learning_rate, 
        learning_rate * 0.01, 
        epochs, 
        len(dataloaders['train']),
        warmup_epochs=warmup_epochs
    )
    
    # Mask ratio schedule (optional progressive masking)
    if mask_ratio_schedule is None:
        # Default: start with less masking, gradually increase
        mask_ratio_schedule = np.linspace(0.5, mask_ratio, epochs)
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_mse': [],
        'val_ssim': [],
        'learning_rate': []
    }
    
    # Training loop
    print(f"Starting training for {epochs} epochs...")
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        # Get current mask ratio
        current_mask_ratio = mask_ratio_schedule[min(epoch, len(mask_ratio_schedule)-1)]
        
        # Training phase
        model.train()
        train_loss = 0
        train_steps = 0
        
        # Initialize GradScaler for mixed precision
        scaler = torch.cuda.amp.GradScaler()
        
        pbar = tqdm(dataloaders['train'], desc=f"Epoch {epoch+1}/{epochs} (Train)")
        for step, batch in enumerate(pbar):
            # Get data
            distance_maps = batch['distance_map'].to(device)
            
            # Update learning rate
            lr = lr_schedule[epoch * len(dataloaders['train']) + step]
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            
            # Forward pass with mixed precision
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                pred, ids_restore, ids_keep = model(distance_maps, mask_ratio=current_mask_ratio)
                # Compute loss in mixed precision
                loss, loss_components = criterion(pred, distance_maps)
            
            # Backward pass and optimizer step with GradScaler
            scaler.scale(loss).backward()
            
            # Gradient clipping (apply before optimizer step)
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            scaler.step(optimizer)
            scaler.update()
            
            # Update metrics
            train_loss += loss.item()
            train_steps += 1
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'lr': f"{lr:.6f}",
                'mask': f"{current_mask_ratio:.2f}"
            })
            
            # Print GPU stats every 100 steps
            if step % 100 == 0:
                print_gpu_utilization()
            
            # Log to wandb
            if use_wandb and step % 10 == 0:
                wandb.log({
                    'train/loss': loss.item(),
                    'train/recon_loss': loss_components['recon'],
                    'train/smooth_loss': loss_components['smooth'],
                    'train/symmetry_loss': loss_components['symmetry'],
                    'train/learning_rate': lr,
                    'train/mask_ratio': current_mask_ratio
                })
        
        avg_train_loss = train_loss / train_steps
        history['train_loss'].append(avg_train_loss)
        history['learning_rate'].append(lr)
        
        # Validation phase
        model.eval()
        val_loss = 0
        val_mse = 0
        val_ssim = 0
        val_steps = 0
        val_samples = 0
        
        with torch.no_grad():
            pbar = tqdm(dataloaders['val'], desc=f"Epoch {epoch+1}/{epochs} (Val)")
            for batch in pbar:
                distance_maps = batch['distance_map'].to(device)
                
                # Forward pass with fixed mask ratio
                # Note: We still pass mask_ratio but it's not used in model.eval()
                pred, _, _, _ = model(distance_maps, mask_ratio=mask_ratio, return_attention=False) # Ensure no attention calculation during regular eval
                
                # Compute loss
                loss, _ = criterion(pred, distance_maps)
                val_loss += loss.item()
                
                # Compute MSE
                mse = nn.functional.mse_loss(pred, distance_maps).item()
                val_mse += mse
                
                # Compute SSIM
                for i in range(distance_maps.size(0)):
                    original = distance_maps[i, 0].cpu().numpy()
                    recon = pred[i, 0].cpu().numpy()
                    
                    if not np.isnan(original).any() and not np.isnan(recon).any():
                        sim = ssim(original, recon, data_range=1.0)
                        val_ssim += sim
                        val_samples += 1
                
                val_steps += 1
                
                pbar.set_postfix({
                    'val_loss': f"{loss.item():.4f}",
                    'val_mse': f"{mse:.4f}"
                })
        
        # Calculate averages
        avg_val_loss = val_loss / val_steps
        avg_val_mse = val_mse / val_steps
        avg_val_ssim = val_ssim / val_samples if val_samples > 0 else 0
        
        history['val_loss'].append(avg_val_loss)
        history['val_mse'].append(avg_val_mse)
        history['val_ssim'].append(avg_val_ssim)
        
        # Print epoch summary
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Val Loss: {avg_val_loss:.4f}")
        print(f"  Val MSE: {avg_val_mse:.4f}")
        print(f"  Val SSIM: {avg_val_ssim:.4f}")
        print(f"  LR: {lr:.6f}, Mask Ratio: {current_mask_ratio:.2f}")
        
        # Log to wandb
        if use_wandb:
            wandb.log({
                'epoch': epoch + 1,
                'val/loss': avg_val_loss,
                'val/mse': avg_val_mse,
                'val/ssim': avg_val_ssim
            })
        
        # Save checkpoint
        if avg_val_loss < best_val_loss: # Changed from loss to avg_val_loss
            best_val_loss = avg_val_loss
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'val_mse': avg_val_mse,
                'val_ssim': avg_val_ssim,
                'mask_ratio': current_mask_ratio
            }
            torch.save(checkpoint, os.path.join(output_dir, 'best_model.pth'))
            print(f"  -> Best model saved!")
        
        # Regular checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            checkpoint_path = os.path.join(output_dir, f'checkpoint_epoch_{epoch+1}.pth')
            torch.save(checkpoint, checkpoint_path)
    
    # Save final model
    final_checkpoint = {
        'epoch': epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'history': history
    }
    torch.save(final_checkpoint, os.path.join(output_dir, 'final_model.pth'))
    
    # Plot training history
    plot_training_history(history, output_dir)
    
    # Evaluate on test set (reconstruction task)
    print("\nEvaluating on test set (reconstruction task)...")
    test_metrics = evaluate_mae(model, dataloaders['test'], device, criterion, 
                               output_dir, mask_ratio=mask_ratio)
    
    print(f"\nTest Results (Reconstruction Task):")
    print(f"  MSE: {test_metrics['mse']:.6f}")
    print(f"  SSIM: {test_metrics['ssim']:.4f}")
    print(f"  Reconstruction Time: {test_metrics['time_per_sample']:.2f} ms")
    
    # Save test metrics
    with open(os.path.join(output_dir, 'test_metrics_recon.txt'), 'w') as f: # Renamed file to specify reconstruction
        for key, value in test_metrics.items():
            if key not in ['sample_originals', 'sample_reconstructions', 'all_cls_tokens', 'collected_attention']:# Exclude collected data for visualization
                f.write(f"{key}: {value}\n")
    
    if use_wandb:
        wandb.finish()
    
    return model, history, test_metrics

def evaluate_mae(model, dataloader, device, criterion, output_dir, mask_ratio=0.75):
    """Evaluate MAE model on test set for reconstruction task."""
    model.eval()
    
    total_mse = 0
    total_ssim = 0
    total_loss = 0
    batch_count = 0
    sample_count = 0
    
    # For timing
    total_time = 0
    time_samples = 0
    
    # For visualization - these will now be collected by the analyze script
    # sample_originals = []
    # sample_reconstructions = []
    # sample_masked = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Evaluating Reconstruction")):
            distance_maps = batch['distance_map'].to(device)
            
            # Time reconstruction
            if time_samples < 50: # Limit timing to avoid slowing down evaluation too much
                start_time = time.time()
                # Call the full model forward - it handles passing to encoder/decoder
                # For evaluation, we don't need attention here, so return_attention=False
                pred, _, _, _ = model(distance_maps, mask_ratio=mask_ratio, return_attention=False)
                end_time = time.time()
                
                batch_time = (end_time - start_time) * 1000  # ms
                total_time += batch_time
                time_samples += 1
            else:
                 # Call the full model forward (no timing after 50 batches)
                 pred, _, _, _ = model(distance_maps, mask_ratio=mask_ratio, return_attention=False)

            # Calculate loss (optional for evaluation, but good to have)
            loss, _ = criterion(pred, distance_maps) # Use the reconstruction loss
            total_loss += loss.item()

            # Calculate MSE
            mse = nn.functional.mse_loss(pred, distance_maps).item()
            total_mse += mse
            batch_count += 1
            
            # Calculate SSIM
            for i in range(distance_maps.size(0)):
                original = distance_maps[i, 0].cpu().numpy()
                recon = pred[i, 0].cpu().numpy()
                
                if not np.isnan(original).any() and not np.isnan(recon).any():
                    sim = ssim(original, recon, data_range=1.0) # Use data_range=1.0 assuming normalized inputs
                    total_ssim += sim
                    sample_count += 1
            
            # Visualization samples will be collected by the separate analysis script
            # Do not collect here to keep evaluation focused and faster

    # Calculate averages
    avg_mse = total_mse / batch_count
    avg_ssim = total_ssim / sample_count if sample_count > 0 else 0 # Avoid division by zero
    avg_loss = total_loss / batch_count
    time_per_sample = total_time / time_samples / dataloader.batch_size if time_samples > 0 and dataloader.batch_size > 0 else 0
    
    # Reconstruction visualization will be done by the separate analysis script
    # Do not call visualize_mae_results here
    
    return {
        'mse': avg_mse,
        'ssim': avg_ssim,
        'loss': avg_loss,
        'time_per_sample': time_per_sample,
        # Do not return visualization data here
    }

def visualize_mae_results(originals, masked, reconstructions, save_path):
    """Visualize MAE results with original, masked, and reconstructed images"""
    n_samples = min(5, len(originals))
    fig, axes = plt.subplots(n_samples, 3, figsize=(12, 4*n_samples)) # Removed masked column
    
    if n_samples == 1:
        axes = axes.reshape(1, -1) # Ensure axes is 2D even for 1 sample
    
    for i in range(n_samples):
        # Original
        im1 = axes[i, 0].imshow(originals[i, 0], cmap='viridis', vmin=0, vmax=1)
        axes[i, 0].set_title('Original')
        axes[i, 0].axis('off')
        
        # Reconstruction
        im2 = axes[i, 1].imshow(reconstructions[i, 0], cmap='viridis', vmin=0, vmax=1)
        axes[i, 1].set_title('Reconstruction')
        axes[i, 1].axis('off')
        
        # Error map
        error = np.abs(originals[i, 0].numpy() - reconstructions[i, 0].numpy())
        im3 = axes[i, 2].imshow(error, cmap='hot', vmin=0, vmax=0.5) # Changed im4 to im3
        axes[i, 2].set_title(f'Error (Mean: {np.mean(error):.4f})') # Changed im4 to im3
        axes[i, 2].axis('off') # Changed im4 to im3
        
        # Add colorbars
        if i == 0:
            fig.colorbar(im1, ax=axes[i, :2], fraction=0.046, pad=0.04) # Adjusted for 2 columns
            fig.colorbar(im3, ax=axes[i, 2], fraction=0.046, pad=0.04) # Adjusted for 1 column
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Visualization saved to {save_path}")

def plot_training_history(history, output_dir):
    """Plot training history"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    # Loss plot
    ax1.plot(history['train_loss'], label='Train Loss', linewidth=2)
    ax1.plot(history['val_loss'], label='Val Loss', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # MSE plot
    ax2.plot(history['val_mse'], label='Val MSE', color='green', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('MSE')
    ax2.set_title('Validation MSE')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # SSIM plot
    ax3.plot(history['val_ssim'], label='Val SSIM', color='orange', linewidth=2)
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('SSIM')
    ax3.set_title('Validation SSIM')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Learning rate plot
    ax4.plot(history['learning_rate'], label='Learning Rate', color='red', linewidth=2)
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Learning Rate')
    ax4.set_title('Learning Rate Schedule')
    ax4.set_yscale('log')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_history.png'), dpi=300)
    plt.close()
    print(f"Training history plot saved to {os.path.join(output_dir, 'training_history.png')}")

def compare_with_baselines(mae_metrics, output_dir):
    """Compare MAE results with CNN and ViT baselines"""
    # Baseline results from your milestone
    baseline_results = {
        'Model': ['Standard CNN', 'Masked CNN (75%)', 'Vision Transformer', 'Protein MAE (Ours)'],
        'MSE': [0.000214, 0.000575, 0.125557, mae_metrics['mse']],
        'SSIM': [0.983256, 0.959629, 0.009061, mae_metrics['ssim']],
        'Time (ms)': [2.81, 0.93, 51.39, mae_metrics['time_per_sample']],
        'Parameters (M)': [0.27, 0.27, 105.10, 2.5]  # Approximate for MAE
    }
    
    # Create comparison plot
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    models = baseline_results['Model']
    x = np.arange(len(models))
    
    # MSE comparison (log scale)
    ax1.bar(x, baseline_results['MSE'])
    ax1.set_yscale('log')
    ax1.set_xlabel('Model')
    ax1.set_ylabel('MSE (log scale)')
    ax1.set_title('MSE Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, rotation=45, ha='right')
    
    # SSIM comparison
    ax2.bar(x, baseline_results['SSIM'])
    ax2.set_xlabel('Model')
    ax2.set_ylabel('SSIM')
    ax2.set_title('SSIM Comparison')
    ax2.set_xticks(x)
    ax2.set_xticklabels(models, rotation=45, ha='right')
    ax2.set_ylim(0, 1)
    
    # Time comparison
    ax3.bar(x, baseline_results['Time (ms)'])
    ax3.set_xlabel('Model')
    ax3.set_ylabel('Time per sample (ms)')
    ax3.set_title('Inference Time Comparison')
    ax3.set_xticks(x)
    ax3.set_xticklabels(models, rotation=45, ha='right')
    
    # Parameter count
    ax4.bar(x, baseline_results['Parameters (M)'])
    ax4.set_xlabel('Model')
    ax4.set_ylabel('Parameters (Millions)')
    ax4.set_title('Model Size Comparison')
    ax4.set_xticks(x)
    ax4.set_xticklabels(models, rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'model_comparison.png'), dpi=300)
    plt.close()
    
    # Save comparison table
    with open(os.path.join(output_dir, 'model_comparison.txt'), 'w') as f:
        f.write("Model Comparison Results\n")
        f.write("=" * 60 + "\n")
        f.write(f"{'Model':<25} {'MSE':<12} {'SSIM':<12} {'Time (ms)':<12} {'Params (M)':<12}\n")
        f.write("-" * 60 + "\n")
        for i in range(len(models)):
            f.write(f"{models[i]:<25} {baseline_results['MSE'][i]:<12.6f} "
                   f"{baseline_results['SSIM'][i]:<12.6f} {baseline_results['Time (ms)'][i]:<12.2f} "
                   f"{baseline_results['Parameters (M)'][i]:<12.2f}\n")

if __name__ == "__main__":
    # Training configuration
    config = {
        'data_dir': "./distance_maps",
        'output_dir': "./mae_results",
        'epochs': 100,
        'batch_size': 256,         # Increased aggressively for T4 optimization
        'learning_rate': 1e-4,
        'mask_ratio': 0.75,
        'warmup_epochs': 2,
        'use_wandb': False,
        'num_workers': 8,
        'pin_memory': True
    }
    
    # --- Uncomment ONE of the following blocks --- #

    # Option 1: Train the model
    model, history, test_metrics = train_protein_mae(**config)
    compare_with_baselines(test_metrics, config['output_dir'])
    
    # Option 2: Analyze a trained model checkpoint
    # checkpoint_path = os.path.join(config['output_dir'], 'best_model.pth') # Or 'final_model.pth'
    # if os.path.exists(checkpoint_path):
    #     analyze_model_visualizations(checkpoint_path, **config) # Pass config to reuse settings
    # else:
    #     print(f"Error: Checkpoint not found at {checkpoint_path}")
    
    print("\nScript execution complete!")