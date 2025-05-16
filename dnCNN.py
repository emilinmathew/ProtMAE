import torch
import torch.nn as nn
from protein_fragment_class import get_dataloaders
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
from torchmetrics.functional import structural_similarity_index_measure as ssim  # Import SSIM from torchmetrics


class DnCNN(nn.Module):
    def __init__(self, in_channels=1, num_layers=17, num_features=64):
        super(DnCNN, self).__init__()
        layers = []
        
        # First layer
        layers.append(nn.Conv2d(in_channels, num_features, kernel_size=3, padding=1, bias=False))
        layers.append(nn.ReLU(inplace=True))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            layers.append(nn.Conv2d(num_features, num_features, kernel_size=3, padding=1, bias=False))
            layers.append(nn.BatchNorm2d(num_features))
            layers.append(nn.ReLU(inplace=True))
        
        # Last layer
        layers.append(nn.Conv2d(num_features, in_channels, kernel_size=3, padding=1, bias=False))
        
        self.dncnn = nn.Sequential(*layers)

    def forward(self, x):
        return x - self.dncnn(x)  # Residual learning

def evaluate_dncnn_model():
    """Evaluate DnCNN model on our protein distance map dataset"""
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_dir = './benchmark_results_dncnn'
    os.makedirs(output_dir, exist_ok=True)
    
    # Create model
    model = DnCNN().to(device)
    
    # Get dataloaders
    dataloaders = get_dataloaders(
        './distance_maps',
        batch_size=100,
        num_workers=4
    )
    
    # Evaluation metrics
    mse_criterion = nn.MSELoss()
    
    # Evaluate on test set
    model.eval()
    test_mse_losses = []
    test_ssim_losses = []
    
    print("Evaluating DnCNN model...")
    with torch.no_grad():
        for batch in tqdm(dataloaders['test']):
            distance_maps = batch['distance_map'].to(device)
            reconstructions = model(distance_maps)
            
            # Compute MSE loss
            mse_loss = mse_criterion(reconstructions, distance_maps)
            test_mse_losses.append(mse_loss.item())
            
            # Compute SSIM loss
            ssim_loss = 1 - ssim(reconstructions, distance_maps)  # SSIM returns similarity, so subtract from 1
            test_ssim_losses.append(ssim_loss.item())
    
    avg_mse_loss = sum(test_mse_losses) / len(test_mse_losses)
    avg_ssim_loss = sum(test_ssim_losses) / len(test_ssim_losses)
    print(f"Average Test MSE Loss: {avg_mse_loss:.4f}")
    print(f"Average Test SSIM Loss: {avg_ssim_loss:.4f}")
    
    # Visualize some reconstructions
    visualize_reconstructions(model, dataloaders['test'], device, output_dir, mse_criterion)
    
    return avg_mse_loss, avg_ssim_loss

def visualize_reconstructions(model, dataloader, device, output_dir, mse_criterion, num_samples=5):
    """Visualize original and reconstructed distance maps with MSE and SSIM loss"""
    
    model.eval()
    batch = next(iter(dataloader))
    distance_maps = batch['distance_map'].to(device)[:num_samples]
    
    with torch.no_grad():
        reconstructions = model(distance_maps)
        mse_losses = [mse_criterion(reconstructions[i:i+1], distance_maps[i:i+1]).item() for i in range(num_samples)]
        ssim_losses = [1 - ssim(reconstructions[i:i+1], distance_maps[i:i+1]).item() for i in range(num_samples)]
    
    # Create visualization
    fig, axes = plt.subplots(num_samples, 2, figsize=(8, 2*num_samples))
    
    for i in range(num_samples):
        # Original
        axes[i, 0].imshow(distance_maps[i, 0].cpu(), cmap='viridis')
        axes[i, 0].set_title('Original')
        axes[i, 0].axis('off')
        
        # Reconstruction
        axes[i, 1].imshow(reconstructions[i, 0].cpu(), cmap='viridis')
        axes[i, 1].set_title(f'Reconstructed\nMSE: {mse_losses[i]:.4f}, SSIM: {1 - ssim_losses[i]:.4f}')
        axes[i, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'dncnn_reconstructions.png'))
    plt.close()

if __name__ == '__main__':
    evaluate_dncnn_model()