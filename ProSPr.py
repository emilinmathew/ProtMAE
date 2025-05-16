from prospr.nn import load_model

# Load the pre-trained ProSPr model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = load_model().to(device)
model.eval()  # Set the model to evaluation mode
print("ProSPr model loaded successfully.")


from prospr.dataloader import get_tensors

def preprocess_distance_maps(distance_map):
    """
    Preprocess a distance map to match ProSPr's input format.
    Args:
        distance_map (numpy.ndarray): The distance map to preprocess.
    Returns:
        torch.Tensor: The preprocessed tensor.
    """
    # Convert the distance map to the required tensor format
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
    """
    Evaluate the ProSPr model on a dataset of distance maps.
    Args:
        distance_maps (list of numpy.ndarray): List of distance maps to evaluate.
        output_dir (str): Directory to save the evaluation results.
    """
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the ProSPr model
    model = load_model().to(device)
    model.eval()
    print("ProSPr model loaded successfully.")
    
    # Evaluation metrics
    criterion = torch.nn.MSELoss()
    test_losses = []
    
    print("Evaluating ProSPr model...")
    with torch.no_grad():
        for i, distance_map in enumerate(tqdm(distance_maps)):
            # Preprocess the distance map
            tensors = get_tensors(distance_map)
            input_tensor = tensors['input'].to(device)  # Input tensor
            target_tensor = tensors['target'].to(device)  # Target tensor
            
            # Forward pass
            output = model(input_tensor)
            
            # Compute loss
            loss = criterion(output, target_tensor)
            test_losses.append(loss.item())
            
            # Save visualization for the first few samples
            if i < 5:
                visualize_reconstruction(output, target_tensor, i, output_dir)
    
    avg_test_loss = sum(test_losses) / len(test_losses)
    print(f"Average Test Loss: {avg_test_loss:.4f}")

def visualize_reconstruction(output, target, idx, output_dir):
    """
    Visualize the original and reconstructed distance maps.
    Args:
        output (torch.Tensor): The reconstructed distance map.
        target (torch.Tensor): The original distance map.
        idx (int): Index of the sample.
        output_dir (str): Directory to save the visualization.
    """
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

# Example usage
if __name__ == '__main__':
    # Example: Load your dataset of distance maps
    distance_maps = [np.random.rand(64, 64) for _ in range(10)]  # Replace with your actual dataset
    
    # Evaluate the ProSPr model
    evaluate_prospr_model(distance_maps)