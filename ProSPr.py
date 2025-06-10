from prospr.nn import ProsprNetwork, load_model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ProsprNetwork().to(device) 

weights_path = './prospr/pretrained_weights.pth' 

load_model(model, weights_path)
model.eval()  
print("ProSPr model loaded successfully.")

from prospr.dataloader import get_tensors

def preprocess_distance_maps(distance_map):
    tensors = get_tensors(distance_map)
    return tensors


    import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
from prospr.nn import load_model
from prospr.dataloader import get_tensors

def evaluate_prospr_model(distance_maps, output_dir='./benchmark_results_prospr'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(output_dir, exist_ok=True)

    model = load_model().to(device)
    model.eval()
    print("ProSPr model loaded successfully.")
    criterion = torch.nn.MSELoss()
    test_losses = []
    
    print("Evaluating ProSPr model...")
    with torch.no_grad():
        for i, distance_map in enumerate(tqdm(distance_maps)):
            #preprocessing for the dist maps 
            tensors = get_tensors(distance_map)
            input_tensor = tensors['input'].to(device) 
            target_tensor = tensors['target'].to(device)  
            
            #forwad pass: 
            output = model(input_tensor)
            
            #loss calc: 
            loss = criterion(output, target_tensor)
            test_losses.append(loss.item())

            if i < 5:
                visualize_reconstruction(output, target_tensor, i, output_dir)
    
    avg_test_loss = sum(test_losses) / len(test_losses)
    print(f"Average Test Loss: {avg_test_loss:.4f}")

def visualize_reconstruction(output, target, idx, output_dir):
    output = output.cpu().numpy()
    target = target.cpu().numpy()
    
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(target[0], cmap='viridis')
    axes[0].set_title('Original')
    axes[0].axis('off')
    
    axes[1].imshow(output[0], cmap='viridis')
    axes[1].set_title('Reconstructed')
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'reconstruction_{idx}.png'))
    plt.close()

if __name__ == '__main__':
    distance_maps = [np.random.rand(64, 64) for _ in range(10)]  
    evaluate_prospr_model(distance_maps)
