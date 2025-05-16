# dataset.py
# Dataset loader for protein fragment distance maps

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
    def __init__(self, root_dir, split='train', transform=None, visualize_samples=False, split_ratios=(0.8, 0.1, 0.1)):
        """
        Args:
            root_dir (str): Directory with the .npz files.
            split (str): 'train', 'val', or 'test'.
            transform (callable, optional): Optional transform to be applied on a sample.
            visualize_samples (bool): Whether to visualize some samples for debugging.
            split_ratios (tuple): Ratios for train, val, and test splits.
        """
        self.root_dir = root_dir
        self.split = split  # Store the split as an instance attribute
        self.transform = transform
        self.visualize_samples = visualize_samples

        # List all .npz files in the directory
        all_files = sorted([str(f) for f in Path(root_dir).glob("map_*.npz")])

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

        print(f"Found {len(self.files)} samples for {split} split")

        # Visualize a few samples if requested
        if visualize_samples:
            self.visualize_random_samples(5)


    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        """
        Args:
            idx (int): Index
            
        Returns:
            dict: A dictionary containing:
                'distance_map': The original distance map
        """
        # Load the .npz file
        file_path = self.files[idx]
        data = np.load(file_path)

        # Extract distance map
        distance_map = data['distance_map'].astype(np.float32)

        # Add channel dimension if missing
        if len(distance_map.shape) == 2:
            distance_map = distance_map[np.newaxis, :, :]

        # Convert to torch tensor
        distance_map = torch.from_numpy(distance_map)

        # Apply transforms if specified
        if self.transform:
            distance_map = self.transform(distance_map)

        # Extract PDB ID from the filename
        pdb_id = Path(file_path).stem.split('_')[1]

        # Create sample dictionary
        sample = {
            'distance_map': distance_map,
            'pdb_id': pdb_id
        }

        return sample

    def visualize_random_samples(self, num_samples=5):
        """
        Visualize random samples from the dataset for debugging.
        """
        if len(self.files) == 0:
            print("No valid samples to visualize")
            return

        indices = np.random.choice(len(self.files), size=min(num_samples, len(self.files)), replace=False)

        fig, axes = plt.subplots(num_samples, 1, figsize=(10, 2.5 * num_samples))

        for i, idx in enumerate(indices):
            sample = self[idx]
            distance_map = sample['distance_map'][0].numpy()  # Remove channel dim

            axes[i].imshow(distance_map, cmap='viridis')
            axes[i].set_title(f"Sample {i + 1} - PDB ID: {sample['pdb_id']}")

        plt.tight_layout()
        plt.savefig(os.path.join(self.root_dir, f"{self.split}_samples.png"))
        plt.close()
        

def get_dataloaders(root_dir, batch_size=512, num_workers=None, visualize_samples=True):
    """
    Create dataloaders for training, validation, and testing.
    
    Args:
        root_dir (str): Directory with the dataset.
        batch_size (int): Batch size.
        num_workers (int): Number of workers for data loading. Defaults to the number of CPU cores.
        visualize_samples (bool): Whether to visualize some samples for debugging.
        
    Returns:
        dict: A dictionary containing dataloaders for 'train', 'val', and 'test' splits.
    """
    if num_workers is None:
        num_workers = os.cpu_count()  # Use all available CPU cores

    # Create datasets
    train_dataset = ProteinFragmentDataset(root_dir, split='train', visualize_samples=visualize_samples)
    val_dataset = ProteinFragmentDataset(root_dir, split='val', visualize_samples=visualize_samples)
    test_dataset = ProteinFragmentDataset(root_dir, split='test', visualize_samples=visualize_samples)
    
    # Debugging: Print dataset sizes
    print(f"Train dataset size: {len(train_dataset)} samples")
    print(f"Validation dataset size: {len(val_dataset)} samples")
    print(f"Test dataset size: {len(test_dataset)} samples")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=4,
        persistent_workers=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=4,
        persistent_workers=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=4,
        persistent_workers=True
    )
    
    return {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader
    }