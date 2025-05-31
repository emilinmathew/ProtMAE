import torch
import torch.nn as nn
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys
from torch.utils.data import DataLoader
# Add parent directory to Python path to import protein_fragment_class
sys.path.append('..') 
from protein_fragment_class import get_dataloaders, ProteinFragmentDataset

# Import the model architecture
from custom_Vit import ProteinDistanceMAE

# Try importing scikit-learn for t-SNE, provide instructions if not found
try:
    from sklearn.manifold import TSNE
except ImportError:
    TSNE = None
    print("Warning: scikit-learn not found. t-SNE visualization will be skipped.")
    print("Please install it: pip install scikit-learn")


# Function to analyze a trained model (load from checkpoint and visualize)
def analyze_model_visualizations(
    checkpoint_path,
    data_dir="../new_distance_maps",
    output_dir="./mae_results",
    batch_size=128, # Use a reasonable batch size for analysis
    num_workers=8,
    pin_memory=True,
    num_viz_samples=10 # Number of samples to use for visualization
):
    """Loads a trained MAE model and generates visualizations (embeddings, attention)."""

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device} for model analysis")

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Load the trained MAE model checkpoint
    print(f"Loading MAE model from {checkpoint_path}...")
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint file not found at {checkpoint_path}")
        return

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

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

    # Prepare data for analysis (using the test set)
    print(f"Loading test dataset from {data_dir} using split files...")
    
    # --- Directly load the test split dataset and dataloader ---
    # Build the full split_files dictionary, assuming splits dir is relative to project root
    splits_dir = './splits'
    split_files = {
        'train': os.path.join(splits_dir, 'train.txt'),
        'val': os.path.join(splits_dir, 'val.txt'),
        'test': os.path.join(splits_dir, 'test.txt')
    }

    # Check if the test split file exists before proceeding
    if not os.path.exists(split_files['test']):
        print(f"Error: Test split file not found at {split_files['test']}.")
        print("Please ensure train_custon_vit.py has been run at least once to generate split files.")
        return

    test_dataset = ProteinFragmentDataset(
        root_dir=data_dir,
        split='test',
        visualize_samples=False, # Don't visualize samples during analysis data loading
        split_files=split_files # Provide the full split_files dictionary
    )

    # Use a smaller batch size for analysis data loading if needed, capped by dataset size
    analysis_batch_size = min(batch_size, len(test_dataset), num_viz_samples * 2) # Cap batch size and num_viz_samples
    
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=analysis_batch_size,
        shuffle=False, # No need to shuffle for analysis
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False # Don't drop last batch for analysis
    )

    print(f"Loaded {len(test_dataset)} samples for analysis.") # Print actual dataset size

    # Collect embeddings and attention weights
    collected_cls_tokens = []
    collected_attention = []
    sample_originals = []
    sample_reconstructions = []

    print(f"Collecting data for visualization from the first {num_viz_samples} samples...")
    with torch.no_grad():
        samples_processed = 0
        for batch in tqdm(test_dataloader, desc="Collecting Viz Data"):
            if samples_processed >= num_viz_samples: # Stop after collecting enough samples
                break

            distance_maps = batch['distance_map'].to(device)

            # Limit batch size to collect only necessary samples from this batch
            current_batch_size = distance_maps.size(0)
            collect_up_to = min(num_viz_samples - samples_processed, current_batch_size)

            if collect_up_to > 0:
                # Get encoder features and attention weights by calling the encoder directly
                # Ensure the encoder is capable of returning attention (check custom_Vit.py)
                # We modified ProteinMAEEncoder to take return_attention=True
                encoder_features, attn_weights_collected, _, _ = model.encoder(
                    distance_maps[:collect_up_to],
                    mask_ratio=0.0, # Don't mask during analysis
                    return_attention=True
                )

                # Collect CLS tokens (features[:, 0])
                collected_cls_tokens.append(encoder_features[:, 0].cpu())
                sample_originals.append(distance_maps[:collect_up_to].cpu())

                # Optionally collect reconstructions - call the full model
                pred, _, _, _ = model(distance_maps[:collect_up_to], mask_ratio=0.0) # No masking for reconstruction analysis
                sample_reconstructions.append(pred.cpu())

                # Collect attention weights: CLS token to other patches from the last block
                if attn_weights_collected is not None:
                    # attn_weights_collected shape: Batch, N_heads, N_tokens, N_tokens
                    # We want the attention from the CLS token (index 0) to all other patches (indices 1 to N-1)
                    collected_attention.append(attn_weights_collected[:, :, 0, 1:].cpu()) # Shape: [batch, N_heads, N_patches]

                samples_processed += collect_up_to

    # Concatenate collected data
    all_cls_tokens = torch.cat(collected_cls_tokens, dim=0) if collected_cls_tokens else None
    all_originals_viz = torch.cat(sample_originals, dim=0) if sample_originals else None
    all_reconstructions_viz = torch.cat(sample_reconstructions, dim=0) if sample_reconstructions else None
    all_attention_viz = torch.cat(collected_attention, dim=0) if collected_attention else None

    # Visualize Reconstructions (can reuse the existing function - copy it here or import if needed)
    # For simplicity, I'll put the visualization functions below analyze_model_visualizations

    # Visualize Embeddings
    if all_cls_tokens is not None and len(all_cls_tokens) > 1 and TSNE is not None:
         visualize_embeddings(all_cls_tokens.numpy(), output_dir)
         
    # Visualize Attention Maps
    if all_attention_viz is not None and all_attention_viz.size(0) > 0:
         # Need patch size and num_heads for attention visualization
         # Get from the model instance
         patch_size = model.encoder.patch_embed.patch_size
         # Check if model.encoder.blocks exists and has an attn attribute
         if hasattr(model.encoder, 'blocks') and len(model.encoder.blocks) > 0 and hasattr(model.encoder.blocks[0], 'attn'):
             num_heads = model.encoder.blocks[0].attn.num_heads # Assuming all blocks have same num_heads
             visualize_attention_maps(all_attention_viz.numpy(), patch_size, num_heads, output_dir)
         else:
             print("Warning: Could not determine patch size or number of heads from the model for attention visualization.")


    print("Model analysis complete.")


# Visualization function for attention maps
def visualize_attention_maps(attention_weights_batch, patch_size, num_heads, output_dir, num_viz=5):
    """Visualize attention maps for a few samples."""
    print("Visualizing attention maps...")
    # Implement plotting attention from CLS token to patches
    # attention_weights_batch shape: Batch, N_heads, N_patches

    num_samples_to_plot = min(num_viz, attention_weights_batch.shape[0])

    # Assuming image is square, h=w
    # N_patches is (img_size // patch_size)**2
    n_patches_per_side = int(attention_weights_batch.shape[-1]**0.5)
    # img_size = n_patches_per_side * patch_size # Not strictly needed for this plot

    # For each sample, plot attention for each head or an average
    # Let's plot the average attention across heads for simplicity in this template

    fig, axes = plt.subplots(num_samples_to_plot, 1, figsize=(8, 4 * num_samples_to_plot))
    if num_samples_to_plot == 1:
        axes = [axes] # Ensure axes is always iterable

    for i in range(num_samples_to_plot):
        # Average attention over heads
        avg_attn = attention_weights_batch[i].mean(axis=0) # Shape: N_patches

        # Reshape to a grid for visualization
        attn_grid = avg_attn.reshape(n_patches_per_side, n_patches_per_side)

        im = axes[i].imshow(attn_grid, cmap='hot', interpolation='nearest')
        axes[i].set_title(f'Sample {i+1} Avg Attention (CLS to Patches)')
        axes[i].axis('off')

    plt.tight_layout()
    attention_viz_path = os.path.join(output_dir, 'analysis_attention.png')
    plt.savefig(attention_viz_path, dpi=300)
    plt.close()
    print(f"Attention visualization saved to {attention_viz_path}")


# Visualization function for embeddings using t-SNE
# Requires scikit-learn for t-SNE
# Requires matplotlib for plotting
def visualize_embeddings(embeddings, output_dir):
    """Visualize learned embeddings using t-SNE."""
    print("Visualizing embeddings using t-SNE...")
    # TSNE import is handled at the top

    # Reduce dimensionality using t-SNE
    # You might need to tune n_components, perplexity, and n_iter depending on your data
    perplexity_val = min(30, embeddings.shape[0] - 1) # Perplexity must be less than n_samples
    if perplexity_val <= 0:
         print("Not enough samples to run t-SNE (need at least 2).")
         return

    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity_val, n_iter=300)
    embeddings_2d = tsne.fit_transform(embeddings)

    # Plot the embeddings
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], s=10, alpha=0.6)
    ax.set_title('t-SNE Visualization of MAE Encoder CLS Token Embeddings')
    ax.set_xlabel('t-SNE Component 1')
    ax.set_ylabel('t-SNE Component 2')
    ax.grid(True, alpha=0.3)

    embeddings_viz_path = os.path.join(output_dir, 'analysis_embeddings_tsne.png')
    plt.savefig(embeddings_viz_path, dpi=300)
    plt.close()
    print(f"Embeddings visualization saved to {embeddings_viz_path}")


if __name__ == "__main__":
    # Configuration for analysis
    analysis_config = {
        'checkpoint_path': './mae_results/best_model.pth', # Path to your saved checkpoint
        'data_dir': "../new_distance_maps", # Data directory
        'output_dir': "./mae_results/analysis", # Output directory for analysis plots
        'batch_size': 128, # Batch size for data loading during analysis
        'num_workers': 8,  # Number of workers for data loading
        'pin_memory': True,
        'num_viz_samples': 100 # Number of samples to use for visualizations (increase for better t-SNE)
    }

    # Create analysis output directory
    os.makedirs(analysis_config['output_dir'], exist_ok=True)

    # Run the analysis
    analyze_model_visualizations(**analysis_config)

    print("\nDownstream evaluation script finished.")