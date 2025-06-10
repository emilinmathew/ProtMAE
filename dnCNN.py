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
        
        #first layer
        layers.append(nn.Conv2d(in_channels, num_features, kernel_size=3, padding=1, bias=False))
        layers.append(nn.ReLU(inplace=True))
        
        #hidden layers
        for _ in range(num_layers - 2):
            layers.append(nn.Conv2d(num_features, num_features, kernel_size=3, padding=1, bias=False))
            layers.append(nn.BatchNorm2d(num_features))
            layers.append(nn.ReLU(inplace=True))
        
        #last layer
        layers.append(nn.Conv2d(num_features, in_channels, kernel_size=3, padding=1, bias=False))
        
        self.dncnn = nn.Sequential(*layers)

    def forward(self, x):
        return x - self.dncnn(x)  

def evaluate_dncnn_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_dir = './benchmark_results_dncnn'
    os.makedirs(output_dir, exist_ok=True)
    
    #create model
    model = DnCNN().to(device)
    
    dataloaders = get_dataloaders(
        './distance_maps',
        batch_size=100,
        num_workers=4
    )
    
    #eval
    mse_criterion = nn.MSELoss()
    
    #eval on test set
    model.eval()
    test_mse_losses = []
    test_ssim_losses = []
    
    with torch.no_grad():
        for batch in tqdm(dataloaders['test']):
            distance_maps = batch['distance_map'].to(device)
            reconstructions = model(distance_maps)
        
            print("Distance Maps:", distance_maps.min().item(), distance_maps.max().item(), distance_maps.mean().item())
            print("Reconstructions:", reconstructions.min().item(), reconstructions.max().item(), reconstructions.mean().item())

            #mse
            mse_loss = mse_criterion(reconstructions, distance_maps)
            test_mse_losses.append(mse_loss.item())
            
            #ssim
            ssim_loss = 1 - ssim(reconstructions, distance_maps)  # SSIM returns similarity, so subtract from 1
            test_ssim_losses.append(ssim_loss.item())
    
    avg_mse_loss = sum(test_mse_losses) / len(test_mse_losses)
    avg_ssim_loss = sum(test_ssim_losses) / len(test_ssim_losses)
    print(f"Average Test MSE Loss: {avg_mse_loss:.4f}")
    print(f"Average Test SSIM Loss: {avg_ssim_loss:.4f}")
    
    visualize_reconstructions(model, dataloaders['test'], device, output_dir, mse_criterion)
    
    return avg_mse_loss, avg_ssim_loss

def visualize_reconstructions(model, dataloader, device, output_dir, mse_criterion, num_samples=5):
    
    model.eval()
    batch = next(iter(dataloader))
    distance_maps = batch['distance_map'].to(device)[:num_samples]
    
    with torch.no_grad():
        reconstructions = model(distance_maps)
        mse_losses = [mse_criterion(reconstructions[i:i+1], distance_maps[i:i+1]).item() for i in range(num_samples)]
        ssim_losses = [1 - ssim(reconstructions[i:i+1], distance_maps[i:i+1]).item() for i in range(num_samples)]
    
    fig, axes = plt.subplots(num_samples, 2, figsize=(8, 2*num_samples))
    
    for i in range(num_samples):
        axes[i, 0].imshow(distance_maps[i, 0].cpu(), cmap='viridis')
        axes[i, 0].set_title('Original')
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(reconstructions[i, 0].cpu(), cmap='viridis')
        axes[i, 1].set_title(f'Reconstructed\nMSE: {mse_losses[i]:.4f}, SSIM: {1 - ssim_losses[i]:.4f}')
        axes[i, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'dncnn_reconstructions.png'))
    plt.close()

if __name__ == '__main__':
    evaluate_dncnn_model()
