import torch
import numpy as np
import json
import os
from torch.utils.data import Dataset, DataLoader
import re

class ProteinNetDataset(Dataset):
    """Dataset for loading ProteinNet data with secondary structure labels"""
    
    def __init__(self, data_dir='proteinnet_data', split='training_95', max_length=128):
        self.data_dir = data_dir
        self.split = split
        self.max_length = max_length
        
        # Load secondary structure annotations
        print("Loading secondary structure annotations...")
        with open(os.path.join(data_dir, 'dssp_annotations.json'), 'r') as f:
            self.ss_annotations = json.load(f)
        
        # Load ProteinNet records
        print(f"Loading {split} data...")
        self.records = self._load_proteinnet_records()
        
        # Filter records that have both distance maps and SS annotations
        print("Filtering records with both distance maps and SS annotations...")
        self.filtered_records = self._filter_records()
        
        print(f"Loaded {len(self.filtered_records)} protein records")
        
    def _load_proteinnet_records(self):
        """Load ProteinNet records from text files"""
        records = []
        
        # Find the proteinnet files
        casp_dir = None
        for item in os.listdir(self.data_dir):
            if item.startswith('casp') and os.path.isdir(os.path.join(self.data_dir, item)):
                casp_dir = os.path.join(self.data_dir, item)
                break
        
        if not casp_dir:
            raise FileNotFoundError("Could not find CASP directory")
        
        # Look for the split file (without .txt extension)
        split_file_path = os.path.join(casp_dir, self.split)
        
        # Check if the expected split file exists
        if not os.path.exists(split_file_path):
            # List available files in the CASP directory
            available = os.listdir(casp_dir)
            # Filter for files that match the split name (e.g., 'training_95', 'validation', 'testing')
            matching_files = [f for f in available if f == self.split]
            
            if matching_files:
                # If a matching file is found, use it
                split_file_path = os.path.join(casp_dir, matching_files[0])
                print(f"Found matching file: {matching_files[0]}")
            else:
                # If no matching file found, raise error
                raise FileNotFoundError(f"No file found for split '{self.split}' in {casp_dir}")
        
        # Parse the ProteinNet text file
        with open(split_file_path, 'r') as f:
            content = f.read()
        
        # Split into individual protein records
        protein_records = content.split('[ID]')
        
        for record in protein_records[1:]:  # Skip first empty split
            if not record.strip():
                continue
                
            lines = record.strip().split('\n')
            protein_id = lines[0].strip()
            
            # Extract relevant information
            protein_data = {'id': protein_id}
            
            i = 1
            while i < len(lines):
                line = lines[i].strip()
                
                if line == '[PRIMARY]':
                    i += 1
                    protein_data['sequence'] = lines[i].strip()
                elif line == '[TERTIARY]':
                    i += 1
                    coords = []
                    while i < len(lines) and not lines[i].startswith('['):
                        coord_line = lines[i].strip()
                        if coord_line:
                            coords.extend([float(x) for x in coord_line.split()])
                        i += 1
                    protein_data['coordinates'] = coords
                    continue
                elif line == '[MASK]':
                    i += 1
                    protein_data['mask'] = lines[i].strip()
                
                i += 1
            
            if 'sequence' in protein_data:
                records.append(protein_data)
            
            # Limit for quick testing
            if len(records) >= 1000:  # Start with smaller subset
                break
        
        return records
    
    def _filter_records(self):
        """Filter records that have both coordinates and SS annotations"""
        filtered = []
        
        for record in self.records:
            pdb_id = record['id']
            
            # Check if we have SS annotation for this protein
            if pdb_id in self.ss_annotations:
                # Check if we have coordinates to compute distance map
                if 'coordinates' in record and len(record['coordinates']) > 0:
                    seq_len = len(record['sequence'])
                    expected_coords = seq_len * 3  # 3 coords per residue (CA atom)
                    
                    if len(record['coordinates']) >= expected_coords and seq_len <= self.max_length:
                        filtered.append(record)
        
        return filtered
    
    def _coords_to_distance_map(self, coordinates, seq_len):
        """Convert 3D coordinates to distance map"""
        # Reshape coordinates to (seq_len, 3) - assuming CA atoms
        coords = np.array(coordinates[:seq_len*3]).reshape(seq_len, 3)
        
        # Compute pairwise distances
        dist_map = np.zeros((seq_len, seq_len))
        for i in range(seq_len):
            for j in range(seq_len):
                dist_map[i, j] = np.linalg.norm(coords[i] - coords[j])
        
        # Normalize distances (optional - you might want to adjust this)
        dist_map = dist_map / 20.0  # Rough normalization
        dist_map = np.clip(dist_map, 0, 1)
        
        return dist_map
    
    def _pad_or_crop(self, data, target_size):
        """Pad or crop data to target size"""
        if len(data) >= target_size:
            return data[:target_size]
        else:
            if isinstance(data, str):
                return data + 'X' * (target_size - len(data))
            elif isinstance(data, np.ndarray):
                if data.ndim == 2:  # Distance map
                    padded = np.zeros((target_size, target_size))
                    padded[:data.shape[0], :data.shape[1]] = data
                    return padded
                else:  # 1D array
                    padded = np.zeros(target_size)
                    padded[:len(data)] = data
                    return padded
    
    def _ss_to_labels(self, ss_string):
        """Convert secondary structure string to numeric labels"""
        # H = alpha helix (0), E = beta sheet (1), C = coil (2)
        ss_map = {'H': 0, 'E': 1, 'C': 2, 'G': 0, 'I': 0, 'B': 1, 'S': 2, 'T': 2, '-': 2}
        return [ss_map.get(ss, 2) for ss in ss_string]
    
    def __len__(self):
        return len(self.filtered_records)
    
    def __getitem__(self, idx):
        record = self.filtered_records[idx]
        
        # Get sequence and coordinates
        sequence = record['sequence']
        coordinates = record['coordinates']
        seq_len = len(sequence)
        
        # Create distance map
        dist_map = self._coords_to_distance_map(coordinates, seq_len)
        
        # Get secondary structure labels
        pdb_id = record['id']
        ss_info = self.ss_annotations[pdb_id]
        ss_string = ss_info['ss']  # Secondary structure string
        ss_labels = self._ss_to_labels(ss_string)
        
        # Pad/crop to consistent size
        dist_map = self._pad_or_crop(dist_map, 64)  # Match your model's expected input
        ss_labels = self._pad_or_crop(np.array(ss_labels), 64)
        
        # Convert to tensors
        dist_map = torch.FloatTensor(dist_map).unsqueeze(0)  # Add channel dimension
        ss_labels = torch.LongTensor(ss_labels)
        
        # For now, return the global secondary structure class (majority vote)
        # You can modify this to do per-residue prediction instead
        global_ss_label = torch.mode(ss_labels[ss_labels < 3])[0]  # Ignore padding
        
        return {
            'distance_map': dist_map,
            'ss_label': global_ss_label,
            'sequence_length': min(seq_len, 64),
            'protein_id': pdb_id
        }

def create_dataloaders(data_dir='proteinnet_data', batch_size=32):
    """Create train/val dataloaders"""
    
    # Create datasets
    train_dataset = ProteinNetDataset(data_dir, split='training_95')  # 95% sequence identity cutoff
    
    # For validation, we'll use a portion of training data
    # In practice, you'd use validation_* files
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    
    train_subset, val_subset = torch.utils.data.random_split(
        train_dataset, [train_size, val_size]
    )
    
    # Create dataloaders
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    return train_loader, val_loader

# Test the data loading
if __name__ == "__main__":
    # First run the download script if data doesn't exist
    if not os.path.exists('proteinnet_data'):
        print("Data not found. Please run the download script first.")
    else:
        train_loader, val_loader = create_dataloaders(batch_size=4)
        
        print("Testing data loading...")
        for batch in train_loader:
            print(f"Batch shape: {batch['distance_map'].shape}")
            print(f"Labels shape: {batch['ss_label'].shape}")
            print(f"Sample labels: {batch['ss_label']}")
            print(f"Sequence lengths: {batch['sequence_length']}")
            break