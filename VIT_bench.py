import torch
import torch.nn as nn
import timm
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import time
import numpy as np
from skimage.metrics import structural_similarity as ssim
from protein_fragment_class import get_dataloaders
import pandas as pd
from PIL import Image

class PretrainedViTAutoencoder(nn.Module):
    def __init__(self, img_size=64, patch_size=8):
        super().__init__()
        self.encoder = timm.create_model(
            'vit_base_patch8_224_in21k', 
            pretrained=True,
            img_size=img_size,
            in_chans=1 
        )
    
        self.encoder.patch_embed.proj = nn.Conv2d(
            1, self.encoder.embed_dim, 
            kernel_size=patch_size, 
            stride=patch_size
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(self.encoder.embed_dim, img_size * img_size),
            nn.Unflatten(1, (1, img_size, img_size)),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder.forward_features(x)
        x = x[:, 0]  
        x = self.decoder(x)
        return x

def evaluate_vit_benchmark(data_dir, output_dir, batch_size=64):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    os.makedirs(output_dir, exist_ok=True)
    split_files = {
        'train': './splits/train.txt',
        'val': './splits/val.txt',
        'test': './splits/test.txt'
    }
    print("Loading datasets...")
    dataloaders = get_dataloaders(
        data_dir,
        batch_size=batch_size,
        split_files=split_files
    )
    print("Creating ViT model...")
    model = PretrainedViTAutoencoder().to(device)
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")

    criterion = nn.MSELoss()
    print("Evaluating ViT model on test set...")
    test_metrics = evaluate_model(model, dataloaders['test'], device, criterion, output_dir)
    print(f"Test Results - MSE: {test_metrics['mse']:.4f}, SSIM: {test_metrics['ssim']:.4f}, Time per sample: {test_metrics['time_per_sample']:.2f} ms")
    
    with open(os.path.join(output_dir, 'vit_test_metrics.txt'), 'w') as f:
        for key, value in test_metrics.items():
            if key not in ['sample_originals', 'sample_reconstructions']:
                f.write(f"{key}: {value}\n")
    
    return test_metrics

def evaluate_model(model, dataloader, device, criterion, output_dir):
    model.eval()
    total_mse = 0
    total_ssim = 0
    batch_count = 0
    sample_count = 0
    total_time = 0
    time_samples = 0
    
    sample_originals = None
    sample_reconstructions = None
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            distance_maps = batch['distance_map'].to(device)
            if time_samples < 50:  #we measure recon time for the 50 samples 
                start_time = time.time()
                reconstructions = model(distance_maps)
                end_time = time.time()
                
                batch_time = (end_time - start_time) * 1000 
                total_time += batch_time
                time_samples += 1
            else:
                reconstructions = model(distance_maps)
            
            #calculating mse loss 
            loss = criterion(reconstructions, distance_maps)
            total_mse += loss.item()
            batch_count += 1
            
            #calculating ssim 
            for i in range(distance_maps.size(0)):
                original = distance_maps[i, 0].cpu().numpy()
                recon = reconstructions[i, 0].cpu().numpy()
                if not np.isnan(original).any() and not np.isnan(recon).any():
                    sim = ssim(original, recon, data_range=1.0)
                    total_ssim += sim
                    sample_count += 1
            
            if sample_originals is None:
                sample_originals = distance_maps[:5].cpu()
                sample_reconstructions = reconstructions[:5].cpu()
    
    avg_mse = total_mse / batch_count
    avg_ssim = total_ssim / sample_count
    time_per_sample = total_time / time_samples / distance_maps.size(0)  
    
    visualize_reconstructions(sample_originals, sample_reconstructions, 
                            os.path.join(output_dir, 'vit_test_reconstructions.png'))
    
    visualize_error_maps(sample_originals, sample_reconstructions,
                        os.path.join(output_dir, 'vit_test_error_maps.png'))
    
    return {
        'mse': avg_mse,
        'ssim': avg_ssim,
        'time_per_sample': time_per_sample,
        'sample_originals': sample_originals,
        'sample_reconstructions': sample_reconstructions
    }

def visualize_reconstructions(originals, reconstructions, save_path):
    fig, axes = plt.subplots(len(originals), 2, figsize=(10, 3*len(originals)))
    
    for i in range(len(originals)):
        axes[i, 0].imshow(originals[i, 0], cmap='viridis')
        axes[i, 0].set_title('Original')
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(reconstructions[i, 0], cmap='viridis')
        axes[i, 1].set_title('Reconstructed')
        axes[i, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def visualize_error_maps(originals, reconstructions, save_path):
    fig, axes = plt.subplots(len(originals), 3, figsize=(15, 3*len(originals)))
    
    for i in range(len(originals)):
        #og 
        axes[i, 0].imshow(originals[i, 0], cmap='viridis')
        axes[i, 0].set_title('Original')
        axes[i, 0].axis('off')
        
        #recon
        axes[i, 1].imshow(reconstructions[i, 0], cmap='viridis')
        axes[i, 1].set_title('Reconstructed')
        axes[i, 1].axis('off')
        
        #error maps 
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
    vit_output_dir = "./benchmark_results/vit"

    vit_metrics = evaluate_vit_benchmark(
        data_dir=data_dir,
        output_dir=vit_output_dir,
        batch_size=64
    )
    
    print("\nViT Benchmark Complete!")
    print(f"Test MSE: {vit_metrics['mse']:.4f}")
    print(f"Test SSIM: {vit_metrics['ssim']:.4f}")
    print(f"Reconstruction Time: {vit_metrics['time_per_sample']:.2f} ms per sample")
