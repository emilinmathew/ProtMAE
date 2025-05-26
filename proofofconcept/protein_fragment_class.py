# protein_fragment_class.py - Updated for new data format
import os
from pathlib import Path
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

num_workers = os.cpu_count()

class ProteinFragmentDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None, visualize_samples=False, 
                 split_ratios=(0.8, 0.1, 0.1), split_files=None):
        """
        Args:
            root_dir (str): Directory with the .npz files.
            split (str): 'train', 'val', or 'test'.
            transform (callable, optional): Optional transform to be applied on a sample.
            visualize_samples (bool): Whether to visualize some samples for debugging.
            split_ratios (tuple): Ratios for train, val, and test splits.
            split_files (dict): Dictionary with paths to split files.
        """
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.visualize_samples = visualize_samples

        if split_files and all(Path(split_files[s]).exists() for s in ['train', 'val', 'test']):
            # Load split from file
            with open(split_files[split], 'r') as f:
                self.files = [line.strip() for line in f.readlines()]
        else:
            # List all .npz files in the directory
            all_files = sorted([str(f) for f in Path(root_dir).glob("map_*.npz")])
            
            if len(all_files) == 0:
                raise ValueError(f"No .npz files found in {root_dir}")

            # Split the dataset
            train_files, test_files = train_test_split(all_files, test_size=split_ratios[2], random_state=42)
            train_files, val_files = train_test_split(train_files, test_size=split_ratios[1] / (split_ratios[0] + split_ratios[1]), random_state=42)

            if split == 'train':
                self.files = train_files
            elif split == 'val':
                self.files = val_files
            elif split == 'test':
                self.files = test_files
            else:
                raise ValueError(f"Invalid split: {split}. Must be 'train', 'val', or 'test'.")

            # Save splits to disk if split_files is provided
            if split_files:
                Path(split_files['train']).parent.mkdir(parents=True, exist_ok=True)
                for split_name, split_files_list in zip(['train', 'val', 'test'], [train_files, val_files, test_files]):
                    with open(split_files[split_name], 'w') as f:
                        f.writelines(f"{file}\n" for file in split_files_list)

        print(f"Found {len(self.files)} samples for {split} split")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        """
        Args:
            idx (int): Index
            
        Returns:
            dict: A dictionary containing:
                'distance_map': The original distance map
                'pdb_id': PDB ID extracted from filename
        """
        # Load the .npz file
        file_path = self.files[idx]
        data = np.load(file_path)

        # Extract distance map
        distance_map = data['distance_map'].astype(np.float32)

        # Check if normalization is needed
        if distance_map.max() > 1.0:
            # Normalize the distance map to [0, 1]
            distance_map = (distance_map - distance_map.min()) / (distance_map.max() - distance_map.min())

        # Add channel dimension if missing
        if len(distance_map.shape) == 2:
            distance_map = distance_map[np.newaxis, :, :]

        # Convert to torch tensor
        distance_map = torch.from_numpy(distance_map)

        # Apply transforms if specified
        if self.transform:
            distance_map = self.transform(distance_map)

        # Extract PDB ID from the filename (format: map_12345_1ABC_01.npz)
        filename = Path(file_path).stem
        parts = filename.split('_')
        if len(parts) >= 3:
            pdb_id = parts[2]  # Should be the PDB ID
        else:
            pdb_id = filename  # Fallback

        # Create sample dictionary
        sample = {
            'distance_map': distance_map,
            'pdb_id': pdb_id
        }

        return sample

    def visualize_random_samples(self, num_samples=5):
        """Visualize random samples from the dataset for debugging."""
        if len(self.files) == 0:
            print("No valid samples to visualize")
            return

        indices = np.random.choice(len(self.files), size=min(num_samples, len(self.files)), replace=False)

        fig, axes = plt.subplots(1, num_samples, figsize=(3*num_samples, 3))
        if num_samples == 1:
            axes = [axes]

        for i, idx in enumerate(indices):
            sample = self[idx]
            distance_map = sample['distance_map'][0].numpy()  # Remove channel dim

            axes[i].imshow(distance_map, cmap='viridis')
            axes[i].set_title(f"PDB: {sample['pdb_id']}")
            axes[i].axis('off')

        plt.tight_layout()
        plt.savefig(os.path.join(self.root_dir, f"{self.split}_samples.png"))
        plt.close()
        print(f"Sample visualizations saved to {self.root_dir}/{self.split}_samples.png")

def get_dataloaders(root_dir, batch_size=512, num_workers=None, visualize_samples=True, split_files=None):
    """
    Create dataloaders for training, validation, and testing.
    """
    if num_workers is None:
        num_workers = min(8, os.cpu_count())  # Cap at 8 to avoid too many processes

    # Create datasets
    train_dataset = ProteinFragmentDataset(root_dir, split='train', visualize_samples=visualize_samples, split_files=split_files)
    val_dataset = ProteinFragmentDataset(root_dir, split='val', visualize_samples=visualize_samples, split_files=split_files)
    test_dataset = ProteinFragmentDataset(root_dir, split='test', visualize_samples=visualize_samples, split_files=split_files)

    # Debugging: Print dataset sizes
    print(f"Train dataset size: {len(train_dataset)} samples")
    print(f"Validation dataset size: {len(val_dataset)} samples")
    print(f"Test dataset size: {len(test_dataset)} samples")

    # Visualize samples if requested
    if visualize_samples:
        train_dataset.visualize_random_samples()

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True if num_workers > 0 else False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True if num_workers > 0 else False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True if num_workers > 0 else False
    )

    return {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader
    }
