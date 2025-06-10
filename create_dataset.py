from protein_fragment_class import get_dataloaders

if __name__ == "__main__":
    root_dir = "./distance_maps"
    split_files = {
        'train': './splits/train.txt',
        'val': './splits/val.txt',
        'test': './splits/test.txt'
    }

    dataloaders = get_dataloaders(root_dir, batch_size=100, num_workers=4, visualize_samples=True, split_files=split_files)
    train_loader = dataloaders['train']
    val_loader = dataloaders['val']
    test_loader = dataloaders['test']

    for batch in train_loader:
        print(f"Batch distance map shape: {batch['distance_map'].shape}")
        print(f"Batch PDB IDs: {batch['pdb_id']}")
        break
