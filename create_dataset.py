def get_dataloaders(root_dir, batch_size=512, num_workers=None, visualize_samples=True, split_files=None):
    """
    Create dataloaders for training, validation, and testing.

    Args:
        root_dir (str): Directory with the dataset.
        batch_size (int): Batch size.
        num_workers (int): Number of workers for data loading. Defaults to the number of CPU cores.
        visualize_samples (bool): Whether to visualize some samples for debugging.
        split_files (dict): Dictionary with paths to split files (e.g., {'train': 'train.txt', 'val': 'val.txt', 'test': 'test.txt'}).

    Returns:
        dict: A dictionary containing dataloaders for 'train', 'val', and 'test' splits.
    """
    if num_workers is None:
        num_workers = os.cpu_count()  # Use all available CPU cores

    # Create datasets
    train_dataset = ProteinFragmentDataset(root_dir, split='train', visualize_samples=visualize_samples, split_files=split_files)
    val_dataset = ProteinFragmentDataset(root_dir, split='val', visualize_samples=visualize_samples, split_files=split_files)
    test_dataset = ProteinFragmentDataset(root_dir, split='test', visualize_samples=visualize_samples, split_files=split_files)

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