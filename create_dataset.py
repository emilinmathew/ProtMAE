from protein_fragment_class import get_dataloaders

# Directory containing .npz files
root_dir = "./distance_maps"

# Create dataloaders
dataloaders = get_dataloaders(root_dir, batch_size=32, num_workers=4, visualize_samples=True)

# Access train, val, and test dataloaders
train_loader = dataloaders['train']
val_loader = dataloaders['val']
test_loader = dataloaders['test']

# Iterate through the train dataloader
for batch in train_loader:
    print(batch['distance_map'].shape, batch['pdb_id'])