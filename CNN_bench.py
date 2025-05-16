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
from protein_fragment_class import get_dataloaders

class CNNAutoencoder(nn.Module):
    def __init__(self):
        super(CNNAutoencoder, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1),  # 32x32
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            
            nn.Conv2d(16, 32, 3, stride=2, padding=1),  # 16x16
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            
            nn.Conv2d(32, 64, 3, stride=2, padding=1),  # 8x8
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            
            nn.Conv2d(64, 128, 3, stride=2, padding=1),  # 4x4
            nn.BatchNorm2d(128),
            nn.ReLU(True),
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),  # 8x8
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),  # 16x16
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1),  # 32x32
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(16, 1, 4, stride=2, padding=1),  # 64x64
            nn.Sigmoid()  # Output normalized between 0 and 1
        )
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

def train_cnn_benchmark(data_dir, output_dir, epochs=10, batch_size=64, learning_rate=0.001):
    """Train and evaluate CNN benchmark model"""
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Define split files
    split_files = {
        'train': './splits/train.txt',
        'val': './splits/val.txt',
        'test': './splits/test.txt'
    }
    
    # Get dataloaders
    print("Loading datasets...")
    dataloaders = get_dataloaders(data_dir, batch_size=batch_size, split_files=split_files)
    
    # Create model
    model = CNNAutoencoder().to(device)
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    train_losses = []
    val_losses = []
    val_ssim_scores = []
    
    print(f"Starting training for {epochs} epochs...")
    for epoch in range(epochs):
        # Training phase
        model.train()
        epoch_loss = 0
        batch_count = 0
        
        for batch in tqdm(dataloaders['train'], desc=f"Epoch {epoch+1}/{epochs} (Train)"):
            distance_maps = batch['distance_map'].to(device)
            
            # Forward pass
            optimizer.zero_grad()
            reconstructions = model(distance_maps)
            loss = criterion(reconstructions, distance_maps)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            batch_count += 1
        
        avg_train_loss = epoch_loss / batch_count
        train_losses.append(avg_train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0
        val_batch_count = 0
        val_ssim = 0
        val_sample_count = 0
        
        with torch.no_grad():
            for batch in tqdm(dataloaders['val'], desc=f"Epoch {epoch+1}/{epochs} (Val)"):
                distance_maps = batch['distance_map'].to(device)
                reconstructions = model(distance_maps)
                
                # Calculate loss
                loss = criterion(reconstructions, distance_maps)
                val_loss += loss.item()
                val_batch_count += 1
                
                # Calculate SSIM
                for i in range(distance_maps.size(0)):
                    original = distance_maps[i, 0].cpu().numpy()
                    recon = reconstructions[i, 0].cpu().numpy()
                    
                    # Ensure valid input for SSIM
                    if not np.isnan(original).any() and not np.isnan(recon).any():
                        sim = ssim(original, recon, data_range=1.0)
                        val_ssim += sim
                        val_sample_count += 1
        
        avg_val_loss = val_loss / val_batch_count
        avg_val_ssim = val_ssim / val_sample_count
        
        val_losses.append(avg_val_loss)
        val_ssim_scores.append(avg_val_ssim)
        
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val SSIM: {avg_val_ssim:.4f}")
        
        # Save model checkpoint
        if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
            checkpoint_path = os.path.join(output_dir, f'cnn_benchmark_epoch_{epoch+1}.pth')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'val_ssim': avg_val_ssim
            }, checkpoint_path)
            print(f"Checkpoint saved to {checkpoint_path}")
    
    # Save the final model
    torch.save(model.state_dict(), os.path.join(output_dir, 'cnn_benchmark_final.pth'))
    
    # Plot training history
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.title('CNN Benchmark Training Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(val_ssim_scores, label='Validation SSIM', color='green')
    plt.xlabel('Epoch')
    plt.ylabel('SSIM')
    plt.title('CNN Benchmark SSIM')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'cnn_training_history.png'))
    print(f"Training history saved to {os.path.join(output_dir, 'cnn_training_history.png')}")
    
    # Evaluate on test set
    test_metrics = evaluate_model(model, dataloaders['test'], device, criterion, output_dir)
    print(f"Test Results - MSE: {test_metrics['mse']:.4f}, SSIM: {test_metrics['ssim']:.4f}, Time per sample: {test_metrics['time_per_sample']:.2f} ms")
    
    # Save test metrics
    with open(os.path.join(output_dir, 'cnn_test_metrics.txt'), 'w') as f:
        for key, value in test_metrics.items():
            if key not in ['sample_originals', 'sample_reconstructions']:
                f.write(f"{key}: {value}\n")
    
    return test_metrics

def evaluate_model(model, dataloader, device, criterion, output_dir):
    """Evaluate model on a dataset and compute metrics"""
    model.eval()
    total_mse = 0
    total_ssim = 0
    batch_count = 0
    sample_count = 0
    
    # For reconstruction time measurement
    total_time = 0
    time_samples = 0
    
    # For visualization
    sample_originals = None
    sample_reconstructions = None
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            distance_maps = batch['distance_map'].to(device)
            
            # Measure reconstruction time
            if time_samples < 50:  # Measure time for 50 batches
                start_time = time.time()
                reconstructions = model(distance_maps)
                end_time = time.time()
                
                batch_time = (end_time - start_time) * 1000  # Convert to ms
                total_time += batch_time
                time_samples += 1
            else:
                reconstructions = model(distance_maps)
            
            # Calculate MSE
            loss = criterion(reconstructions, distance_maps)
            total_mse += loss.item()
            batch_count += 1
            
            # Calculate SSIM
            for i in range(distance_maps.size(0)):
                original = distance_maps[i, 0].cpu().numpy()
                recon = reconstructions[i, 0].cpu().numpy()
                
                # Ensure valid input for SSIM
                if not np.isnan(original).any() and not np.isnan(recon).any():
                    sim = ssim(original, recon, data_range=1.0)
                    total_ssim += sim
                    sample_count += 1
            
            # Store sample for visualization
            if sample_originals is None:
                sample_originals = distance_maps[:5].cpu()
                sample_reconstructions = reconstructions[:5].cpu()
    
    # Calculate average metrics
    avg_mse = total_mse / batch_count
    avg_ssim = total_ssim / sample_count
    time_per_sample = total_time / time_samples / distance_maps.size(0)  # ms per sample
    
    # Visualize reconstructions
    visualize_reconstructions(sample_originals, sample_reconstructions, 
                            os.path.join(output_dir, 'cnn_test_reconstructions.png'))
    
    # Visualize error maps
    visualize_error_maps(sample_originals, sample_reconstructions,
                        os.path.join(output_dir, 'cnn_test_error_maps.png'))
    
    # Return metrics
    return {
        'mse': avg_mse,
        'ssim': avg_ssim,
        'time_per_sample': time_per_sample,
        'sample_originals': sample_originals,
        'sample_reconstructions': sample_reconstructions
    }

def visualize_reconstructions(originals, reconstructions, save_path):
    """Visualize original and reconstructed distance maps"""
    fig, axes = plt.subplots(len(originals), 2, figsize=(10, 3*len(originals)))
    
    for i in range(len(originals)):
        # Original
        axes[i, 0].imshow(originals[i, 0], cmap='viridis')
        axes[i, 0].set_title('Original')
        axes[i, 0].axis('off')
        
        # Reconstruction
        axes[i, 1].imshow(reconstructions[i, 0], cmap='viridis')
        axes[i, 1].set_title('Reconstructed')
        axes[i, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def visualize_error_maps(originals, reconstructions, save_path):
    """Visualize error maps between originals and reconstructions"""
    fig, axes = plt.subplots(len(originals), 3, figsize=(15, 3*len(originals)))
    
    for i in range(len(originals)):
        # Original
        axes[i, 0].imshow(originals[i, 0], cmap='viridis')
        axes[i, 0].set_title('Original')
        axes[i, 0].axis('off')
        
        # Reconstruction
        axes[i, 1].imshow(reconstructions[i, 0], cmap='viridis')
        axes[i, 1].set_title('Reconstructed')
        axes[i, 1].axis('off')
        
        # Error map
        error = np.abs(originals[i, 0].numpy() - reconstructions[i, 0].numpy())
        im = axes[i, 2].imshow(error, cmap='hot')
        axes[i, 2].set_title(f'Error (Mean: {np.mean(error):.4f})')
        axes[i, 2].axis('off')
        fig.colorbar(im, ax=axes[i, 2], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

if __name__ == "__main__":
    data_dir = "./distance_maps"
    output_dir = "./benchmark_results/cnn"
    
    # Run CNN benchmark
    test_metrics = train_cnn_benchmark(
        data_dir=data_dir,
        output_dir=output_dir,
        epochs=10,
        batch_size=64,
        learning_rate=0.001
    )
    
    print("\nCNN Benchmark Complete!")
    print(f"Test MSE: {test_metrics['mse']:.4f}")
    print(f"Test SSIM: {test_metrics['ssim']:.4f}")
    print(f"Reconstruction Time: {test_metrics['time_per_sample']:.2f} ms per sample")