import torch
import torch.nn as nn
import timm
from protein_fragment_class import get_dataloaders
from tqdm import tqdm
import matplotlib.pyplot as plt
import os

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
            nn.Unflatten(1, (1, img_size, img_size))
        )

    def forward(self, x):
        x = self.encoder.forward_features(x)
        x = x[:, 0]  
        x = self.decoder(x)
        return x

def evaluate_pretrained_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_dir = './benchmark_results'
    os.makedirs(output_dir, exist_ok=True)

    model = PretrainedViTAutoencoder().to(device)
    dataloaders = get_dataloaders(
        './distance_maps',
        batch_size=100,
        num_workers=4
    )
    
    criterion = nn.MSELoss()
    model.eval()
    test_losses = []
    
    print("Evaluating pre-trained ViT model...")
    with torch.no_grad():
        for batch in tqdm(dataloaders['test']):
            distance_maps = batch['distance_map'].to(device)
            reconstructions = model(distance_maps)
            loss = criterion(reconstructions, distance_maps)
            test_losses.append(loss.item())
    
    avg_test_loss = sum(test_losses) / len(test_losses)
    print(f"Average Test Loss: {avg_test_loss:.4f}")
    visualize_reconstructions(model, dataloaders['test'], device, output_dir, criterion)
    
    return avg_test_loss

def visualize_reconstructions(model, dataloader, device, output_dir, criterion, num_samples=5):
    model.eval()
    batch = next(iter(dataloader))
    distance_maps = batch['distance_map'].to(device)[:num_samples]
    
    with torch.no_grad():
        reconstructions = model(distance_maps)
        losses = [criterion(reconstructions[i:i+1], distance_maps[i:i+1]).item() for i in range(num_samples)]

    fig, axes = plt.subplots(num_samples, 2, figsize=(8, 2*num_samples))
    
    for i in range(num_samples):
        axes[i, 0].imshow(distance_maps[i, 0].cpu(), cmap='viridis')
        axes[i, 0].set_title('Original')
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(reconstructions[i, 0].cpu(), cmap='viridis')
        axes[i, 1].set_title(f'Reconstructed (MSE: {losses[i]:.4f})')
        axes[i, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'vit_reconstructions.png'))
    plt.close()

if __name__ == '__main__':
    evaluate_pretrained_model()
