import sys
sys.path.append('..')

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
from custom_Vit import ProteinDistanceMAE  # Your MAE model

class ProteinSSPredictor(nn.Module):
    """Secondary Structure Predictor using your trained MAE encoder"""
    
    def __init__(self, mae_model_path, num_classes=3, freeze_encoder=True):
        super().__init__()
        
        # Load your trained MAE model
        print(f"Loading MAE model from {mae_model_path}")
        self.mae = ProteinDistanceMAE(
            img_size=64, patch_size=4, embed_dim=256, 
            decoder_embed_dim=128, depth=8, decoder_depth=4,
            num_heads=8, decoder_num_heads=8
        )
        
        # Load pretrained weights
        checkpoint = torch.load(mae_model_path, map_location='cpu')
        if 'model_state_dict' in checkpoint:
            self.mae.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.mae.load_state_dict(checkpoint)
        
        # Freeze encoder if specified
        if freeze_encoder:
            for param in self.mae.encoder.parameters():
                param.requires_grad = False
            print("Froze MAE encoder weights")
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),  # Your embed_dim is 256
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )
        
        # Initialize classifier weights
        self._init_classifier_weights()
    
    def _init_classifier_weights(self):
        """Initialize classifier weights"""
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Extract features using MAE encoder (no masking)
        with torch.set_grad_enabled(not hasattr(self, '_freeze_encoder') or not self._freeze_encoder):
            features, _, _, _ = self.mae.encoder(x, mask_ratio=0.0)
        
        # Use CLS token as global protein representation
        cls_token = features[:, 0]  # Shape: (batch_size, embed_dim)
        
        # Classify
        logits = self.classifier(cls_token)
        return logits

class SSPredictionTrainer:
    """Trainer for secondary structure prediction"""
    
    def __init__(self, model, train_loader, val_loader, device='cuda'):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        # Loss and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=1e-4, weight_decay=0.01
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', patience=3, factor=0.5
        )
        
        # Tracking
        self.train_losses = []
        self.val_accuracies = []
        self.best_val_acc = 0.0
    
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        for batch in self.train_loader:
            distance_maps = batch['distance_map'].to(self.device)
            labels = batch['ss_label'].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            logits = self.model(distance_maps)
            loss = self.criterion(logits, labels)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        self.train_losses.append(avg_loss)
        return avg_loss
    
    def validate(self):
        """Validate the model"""
        self.model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in self.val_loader:
                distance_maps = batch['distance_map'].to(self.device)
                labels = batch['ss_label'].to(self.device)
                
                logits = self.model(distance_maps)
                preds = torch.argmax(logits, dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        accuracy = accuracy_score(all_labels, all_preds)
        self.val_accuracies.append(accuracy)
        
        return accuracy, all_preds, all_labels
    
    def train(self, num_epochs=20):
        """Full training loop"""
        print(f"Training for {num_epochs} epochs...")
        
        for epoch in range(num_epochs):
            # Train
            train_loss = self.train_epoch()
            
            # Validate
            val_acc, val_preds, val_labels = self.validate()
            
            # Learning rate scheduling
            self.scheduler.step(val_acc)
            
            # Save best model
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_accuracy': val_acc,
                    'epoch': epoch
                }, 'best_ss_model.pth')
            
            print(f"Epoch {epoch+1}/{num_epochs}: "
                  f"Train Loss: {train_loss:.4f}, "
                  f"Val Acc: {val_acc:.4f}, "
                  f"LR: {self.optimizer.param_groups[0]['lr']:.6f}")
        
        # Final evaluation
        print(f"\nBest validation accuracy: {self.best_val_acc:.4f}")
        
        # Classification report
        val_acc, val_preds, val_labels = self.validate()
        class_names = ['Alpha Helix', 'Beta Sheet', 'Coil']
        print("\nFinal Classification Report:")
        print(classification_report(val_labels, val_preds, target_names=class_names))
        
        return self.best_val_acc
    
    def plot_training_curves(self):
        """Plot training curves"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Training loss
        ax1.plot(self.train_losses)
        ax1.set_title('Training Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.grid(True)
        
        # Validation accuracy
        ax2.plot(self.val_accuracies)
        ax2.set_title('Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig('training_curves.png', dpi=150, bbox_inches='tight')
        plt.show()

def main():
    """Main training function"""
    
    # Check if data exists
    if not os.path.exists('proteinnet_data'):
        print("Please run the download script first to get ProteinNet data!")
        return
    
    # Load data
    from proteinnet_processor import create_dataloaders
    print("Loading data...")
    train_loader, val_loader = create_dataloaders(batch_size=16)
    
    # Initialize model (replace with your actual MAE model path)
    mae_model_path = '../mae_results/best_model.pth'  # UPDATE THIS PATH!
    
    if not os.path.exists(mae_model_path):
        print(f"MAE model not found at {mae_model_path}")
        print("Please update the path to your trained MAE model!")
        return
    
    model = ProteinSSPredictor(mae_model_path, num_classes=3, freeze_encoder=True)
    
    # Check device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Train
    trainer = SSPredictionTrainer(model, train_loader, val_loader, device)
    best_acc = trainer.train(num_epochs=20)
    
    # Plot results
    trainer.plot_training_curves()
    
    print(f"\nTraining completed! Best accuracy: {best_acc:.4f}")
    print("Model saved as 'best_ss_model.pth'")

if __name__ == "__main__":
    import os
    main()