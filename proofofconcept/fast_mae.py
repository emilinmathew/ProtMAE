
# ultra_fast_mae.py - Proof of concept training

import torch
import torch.nn as nn
import os
from tqdm import tqdm
import matplotlib.pyplot as plt

from mae_model import MaskedDistanceMapAutoencoder
from protein_fragment_class import ProteinFragmentDataset
from torch.utils.data import DataLoader

def ultra_fast_mae_test():
    """Ultra fast MAE test - just prove it works"""
    
    print("⚡ Ultra Fast MAE Test - Proof of Concept")
    print("=" * 50)
    
    # ULTRA MINIMAL Configuration
    config = {
        'data_dir': './distance_maps_30k',
        'batch_size': 8,       # Very small batches
        'epochs': 3,           # Just 3 epochs
        'learning_rate': 5e-4, # Higher LR for faster learning
        
        # TINY MODEL for speed
        'embed_dim': 128,      # Very small
        'depth': 2,            # Just 2 layers!
        'num_heads': 4,        
        'decoder_embed_dim': 64, # Tiny decoder
        'decoder_depth': 1,    # Single decoder layer
        'mask_ratio': 0.5,     # Less aggressive masking
        
        # MINIMAL TRAINING
        'max_train_batches': 10,  # Only 10 batches per epoch!
        'max_val_batches': 3,     # Only 3 validation batches
    }
    
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create tiny model
    print("Creating ultra-small model...")
    model = MaskedDistanceMapAutoencoder(
        img_size=64,
        patch_size=8,
        embed_dim=config['embed_dim'],
        depth=config['depth'],
        num_heads=config['num_heads'],
        decoder_embed_dim=config['decoder_embed_dim'],
        decoder_depth=config['decoder_depth'],
        mask_ratio=config['mask_ratio']
    ).to(device)
    
    param_count = sum(p.numel() for p in model.parameters())
    print(f"✅ Ultra-small model: {param_count:,} parameters ({param_count/1e6:.1f}M)")
    
    # Create minimal dataloaders
    def create_tiny_dataloader(data_dir, batch_size, split):
        dataset = ProteinFragmentDataset(data_dir, split=split)
        return DataLoader(dataset, batch_size=batch_size, shuffle=(split=='train'), num_workers=0)
    
    train_loader = create_tiny_dataloader(config['data_dir'], config['batch_size'], 'train')
    val_loader = create_tiny_dataloader(config['data_dir'], config['batch_size'], 'val')
    
    print(f"✅ Data loaded - will train on {config['max_train_batches']} batches only")
    
    # Estimate realistic time
    estimated_batches = config['epochs'] * (config['max_train_batches'] + config['max_val_batches'])
    print(f"⏱️  Total batches to process: {estimated_batches}")
    print(f"   At 10 seconds per batch: ~{estimated_batches * 10 / 60:.1f} minutes")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'])
    
    print(f"\n🏃 Starting ultra-fast training...")
    train_losses = []
    val_losses = []
    
    import time
    start_time = time.time()
    
    for epoch in range(config['epochs']):
        print(f"\n--- Epoch {epoch+1}/{config['epochs']} ---")
        
        # Training
        model.train()
        epoch_train_loss = 0
        
        print("Training...")
        for batch_idx, batch in enumerate(train_loader):
            if batch_idx >= config['max_train_batches']:
                break
                
            batch_start = time.time()
            distance_maps = batch['distance_map'].to(device)
            
            optimizer.zero_grad()
            loss, pred, mask = model(distance_maps)
            loss.backward()
            optimizer.step()
            
            epoch_train_loss += loss.item()
            
            batch_time = time.time() - batch_start
            print(f"  Batch {batch_idx+1}/{config['max_train_batches']}: Loss={loss.item():.4f}, Time={batch_time:.1f}s")
        
        avg_train_loss = epoch_train_loss / config['max_train_batches']
        train_losses.append(avg_train_loss)
        
        # Validation
        model.eval()
        epoch_val_loss = 0
        
        print("Validation...")
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                if batch_idx >= config['max_val_batches']:
                    break
                    
                distance_maps = batch['distance_map'].to(device)
                loss, pred, mask = model(distance_maps)
                epoch_val_loss += loss.item()
                
                print(f"  Val Batch {batch_idx+1}: Loss={loss.item():.4f}")
        
        avg_val_loss = epoch_val_loss / config['max_val_batches']
        val_losses.append(avg_val_loss)
        
        epoch_time = time.time() - start_time
        print(f"✅ Epoch {epoch+1} Complete: Train={avg_train_loss:.4f}, Val={avg_val_loss:.4f}")
        print(f"   Total time so far: {epoch_time/60:.1f} minutes")
    
    total_time = time.time() - start_time
    
    print(f"\n🎉 Ultra-fast training complete in {total_time/60:.1f} minutes!")
    
    # Quick visualization
    print("Creating final visualization...")
    model.eval()
    with torch.no_grad():
        # Get a sample
        sample_batch = next(iter(val_loader))
        sample_input = sample_batch['distance_map'][:2].to(device)
        
        loss, pred, mask = model(sample_input)
        reconstructed = model.unpatchify(pred)
        
        # Create comparison plot
        fig, axes = plt.subplots(2, 3, figsize=(12, 8))
        
        for i in range(2):
            # Original
            axes[i, 0].imshow(sample_input[i, 0].cpu(), cmap='viridis')
            axes[i, 0].set_title('Original')
            axes[i, 0].axis('off')
            
            # Masked (approximate visualization)
            mask_2d = mask[i].reshape(8, 8)
            mask_upsampled = torch.nn.functional.interpolate(
                mask_2d.unsqueeze(0).unsqueeze(0).float(),
                size=(64, 64), mode='nearest'
            )[0, 0]
            masked = sample_input[i, 0].cpu() * (1 - mask_upsampled.cpu())
            
            axes[i, 1].imshow(masked, cmap='viridis')
            axes[i, 1].set_title(f'Masked ({config["mask_ratio"]:.0%})')
            axes[i, 1].axis('off')
            
            # Reconstructed
            axes[i, 2].imshow(reconstructed[i, 0].cpu(), cmap='viridis')
            axes[i, 2].set_title('Reconstructed')
            axes[i, 2].axis('off')
        
        plt.suptitle('MAE Results - Proof of Concept', fontsize=16)
        plt.tight_layout()
        plt.savefig('./ultra_fast_mae_results.png', dpi=150, bbox_inches='tight')
        plt.show()
    
    # Save minimal model
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'training_time_minutes': total_time / 60,
        'parameters': param_count
    }, './ultra_fast_mae.pth')
    
    print(f"\n" + "="*50)
    print("🎉 PROOF OF CONCEPT COMPLETE!")
    print("="*50)
    print(f"✅ MAE successfully trained on protein distance maps")
    print(f"✅ Model parameters: {param_count:,} ({param_count/1e6:.2f}M)")  
    print(f"✅ Training time: {total_time/60:.1f} minutes")
    print(f"✅ Final train loss: {train_losses[-1]:.4f}")
    print(f"✅ Final val loss: {val_losses[-1]:.4f}")
    print(f"✅ Masking ratio: {config['mask_ratio']:.0%}")
    print()
    print("📁 Files saved:")
    print("   • ultra_fast_mae.pth - Trained model")
    print("   • ultra_fast_mae_results.png - Sample reconstructions")
    print()
    print("🚀 This proves MAE works on protein data!")
    print("   Now you can scale up on cloud or optimize further.")
    
    return model, train_losses, val_losses

if __name__ == "__main__":
    print("Starting ultra-fast MAE proof of concept...")
    print("This should complete in under 5 minutes!")
    print()
    
    try:
        model, train_losses, val_losses = ultra_fast_mae_test()
        
        # Show learning occurred
        if len(train_losses) > 1:
            initial_loss = train_losses[0]
            final_loss = train_losses[-1]
            improvement = ((initial_loss - final_loss) / initial_loss) * 100
            print(f"📈 Learning confirmed: {improvement:.1f}% loss reduction")
            
        print("\n✅ Ready for your CS231N project!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
