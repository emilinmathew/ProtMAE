import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys
from torch.utils.data import DataLoader, Subset
# Add parent directory to Python path to import protein_fragment_class
sys.path.append('..') 
from protein_fragment_class import get_dataloaders, ProteinFragmentDataset

# Import the model architecture
from custom_Vit import ProteinDistanceMAE

# Try importing scikit-learn for similarity search, provide instructions if not found
try:
    from sklearn.metrics.pairwise import cosine_similarity # Import cosine_similarity
except ImportError:
    cosine_similarity = None
    print("Warning: scikit-learn not found. Similarity search will be skipped.")
    print("Please install it: pip install scikit-learn")


def find_similar_samples(
    query_sample_index, 
    all_embeddings, 
    dataset, # Pass the full dataset to get original file paths
    top_k=10
):
    """Find samples most similar to a query sample based on embeddings."""
    if all_embeddings is None or len(all_embeddings) == 0 or cosine_similarity is None:
        print("Error: Embeddings not available or scikit-learn not installed for similarity search.")
        return []

    print(f"Finding top {top_k} similar samples to sample {query_sample_index}...")
    
    if query_sample_index < 0 or query_sample_index >= len(all_embeddings):
        print(f"Error: Query sample index {query_sample_index} is out of bounds for the collected embeddings.")
        return []

    # Get the query embedding and add a dimension to make it 2D [1, embed_dim]
    query_embedding = all_embeddings[query_sample_index:query_sample_index+1] # Use slicing to keep dimension

    # Compute cosine similarity between the query and all other embeddings
    # Cosine similarity is a good choice for comparing embeddings
    similarities = cosine_similarity(query_embedding, all_embeddings)[0] # Shape: N_samples

    # Get the indices of the top_k most similar samples (excluding the query itself)
    # Sort in descending order and take top k+1 (to include the query)
    most_similar_indices = np.argsort(similarities)[::-1][:top_k + 1]

    # Filter out the query sample index if it's in the top_k
    most_similar_indices = [idx for idx in most_similar_indices if idx != query_sample_index][:top_k]

    print("Top similar samples (index, similarity score, original file):")
    results = []
    for idx in most_similar_indices:
        similarity_score = similarities[idx]
        # Retrieve original file path from the dataset
        # Note: The index here is relative to the viz_subset_dataset, not the full test_dataset
        # To get the original file path from the full dataset, we need to map back the index
        # The viz_subset_dataset was created from range(min(num_viz_samples, len(test_dataset)))
        # So the index in viz_subset_dataset corresponds directly to the index in test_dataset
        original_file = dataset.samples[idx]['file_path'] if hasattr(dataset, 'samples') and len(dataset.samples) > idx and 'file_path' in dataset.samples[idx] else f"Index {idx}"
        results.append((idx, similarity_score, original_file))
        print(f"  Index: {idx}, Similarity: {similarity_score:.4f}, File: {original_file}")
    
    return results


def visualize_reconstructions(original_images, reconstructed_images, output_dir, num_viz=10):
    """Visualize original and reconstructed distance maps."""
    print("Visualizing reconstructions...")
    
    num_samples_to_plot = min(num_viz, original_images.shape[0])
    
    fig, axes = plt.subplots(num_samples_to_plot, 2, figsize=(8, 4 * num_samples_to_plot))
    if num_samples_to_plot == 1:
        axes = [axes] # Ensure axes is always iterable

    for i in range(num_samples_to_plot):
        # Original
        im_orig = axes[i, 0].imshow(original_images[i, 0].squeeze().numpy(), cmap='viridis', origin='lower')
        axes[i, 0].set_title(f'Sample {i+1} Original')
        axes[i, 0].axis('off')

        # Reconstruction
        im_recon = axes[i, 1].imshow(reconstructed_images[i, 0].squeeze().numpy(), cmap='viridis', origin='lower')
        axes[i, 1].set_title(f'Sample {i+1} Reconstruction')
        axes[i, 1].axis('off')

    plt.tight_layout()
    reconstruction_viz_path = os.path.join(output_dir, 'analysis_reconstructions.png')
    plt.savefig(reconstruction_viz_path, dpi=300)
    plt.close()
    print(f"Reconstruction visualization saved to {reconstruction_viz_path}")


if __name__ == "__main__":
    # Configuration for analysis
    analysis_config = {
        'checkpoint_path': './mae_results/best_model.pth', # Path to your saved checkpoint
        'data_dir': "../new_distance_maps", # Data directory
        'output_dir': "./mae_results/analysis", # Output directory for analysis plots
        'batch_size': 128, # Use a reasonable batch size for analysis (should be > num_viz_samples)
        'num_workers': 8,  # Number of workers for data loading
        'pin_memory': True,
        'num_viz_samples': 100, # Number of samples for Reconstruction/Attention visualizations and Similarity Search
        'query_sample_index': 0, # Index of the sample to use as a query for similarity search (index relative to collected viz samples)
        'top_k_similar': 10 # Number of most similar samples to find
    }

    # Create analysis output directory
    os.makedirs(analysis_config['output_dir'], exist_ok=True)

    # --- Analysis Steps ---

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device} for model analysis")

    # Create output directory if it doesn't exist
    os.makedirs(analysis_config['output_dir'], exist_ok=True)

    # Load the trained MAE model checkpoint
    checkpoint_path = analysis_config['checkpoint_path']
    print(f"Loading MAE model from {checkpoint_path}...")
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint file not found at {checkpoint_path}")
        # return # Don't exit, allow other steps if possible

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False) # weights_only=False for older models

    # Instantiate the full MAE model architecture (decoder parts are not used for analysis,
    # but the full class is needed to load the state_dict)
    model = ProteinDistanceMAE(
        img_size=64,
        patch_size=4,
        embed_dim=256,
        decoder_embed_dim=128,
        depth=8,
        decoder_depth=4,
        num_heads=8,
        decoder_num_heads=8
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval() # Set model to evaluation mode
    print("MAE model loaded successfully.")

    # Prepare data for analysis (using the test set) - Load the full dataset initially
    data_dir = analysis_config['data_dir']
    splits_dir = './splits' # Assuming splits dir is relative to project root
    split_files = {
        'train': os.path.join(splits_dir, 'train.txt'),
        'val': os.path.join(splits_dir, 'val.txt'),
        'test': os.path.join(splits_dir, 'test.txt')
    }

    # Check if the test split file exists before proceeding
    if not os.path.exists(split_files['test']):
        print(f"Error: Test split file not found at {split_files['test']}.")
        print("Please ensure train_custon_vit.py has been run at least once to generate split files.")
        # return # Don't exit, allow other steps if possible

    test_dataset = ProteinFragmentDataset(
        root_dir=data_dir,
        split='test',
        visualize_samples=False, # Don't visualize samples during analysis data loading
        split_files=split_files # Provide the full split_files dictionary
    )
    print(f"Loaded {len(test_dataset)} samples for analysis.") # Print actual dataset size

    # Collect data for Reconstruction Visualization and Similarity Search (limited samples)
    collected_cls_tokens = []
    sample_originals = []
    sample_reconstructions = []

    num_viz_samples = analysis_config['num_viz_samples']
    if num_viz_samples > 0:
        print(f"Collecting data for Visualization and Similarity Search from the first {num_viz_samples} samples...")
        # Create a new dataloader for a small subset
        viz_subset_dataset = torch.utils.data.Subset(test_dataset, range(min(num_viz_samples, len(test_dataset))))
        viz_dataloader = DataLoader(
            viz_subset_dataset,
            batch_size=min(num_viz_samples, len(viz_subset_dataset)), # Use a batch size up to num_viz_samples
            shuffle=False,
            num_workers=analysis_config['num_workers'],
            pin_memory=analysis_config['pin_memory'],
            drop_last=False
        )

        with torch.no_grad():
             # Assuming there's only one batch in viz_dataloader due to batch_size setting
            for batch in tqdm(viz_dataloader, desc="Collecting Viz/Search Samples"):
                distance_maps_viz = batch['distance_map'].to(device)
                sample_originals.append(distance_maps_viz.cpu())

                # Get encoder features for visualization and similarity search
                encoder_features_viz, _, _, _ = model.encoder(
                    distance_maps_viz,
                    mask_ratio=0.0, # Don't mask during analysis
                    return_attention=False # Don't need attention here
                )
                # Collect CLS tokens for similarity search
                collected_cls_tokens.append(encoder_features_viz[:, 0].cpu())

                # Get reconstructions for visualization samples - call the full model
                pred_viz, _, _, _ = model(distance_maps_viz, mask_ratio=0.0)
                sample_reconstructions.append(pred_viz.cpu())

    all_originals = torch.cat(sample_originals, dim=0) if sample_originals else None
    all_reconstructions = torch.cat(sample_reconstructions, dim=0) if sample_reconstructions else None
    all_cls_tokens_subset = torch.cat(collected_cls_tokens, dim=0) if collected_cls_tokens else None

    # --- Perform Similarity Search ---
    if all_cls_tokens_subset is not None:
        find_similar_samples(
            query_sample_index=analysis_config['query_sample_index'],
            all_embeddings=all_cls_tokens_subset.numpy(), # Pass as numpy array for scikit-learn
            dataset=viz_subset_dataset, # Pass the subset dataset to retrieve file paths and map indices correctly
            top_k=analysis_config['top_k_similar']
        )

    # --- Perform Visualizations ---

    # Visualize Reconstructions
    if all_originals is not None and all_reconstructions is not None:
        visualize_reconstructions(all_originals, all_reconstructions, analysis_config['output_dir'], num_viz=num_viz_samples)

    print("Model analysis complete.")

    print("\nDownstream evaluation script finished.")