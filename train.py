"""
Training script for breath sound classification
"""

import os
import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
from datetime import datetime

from src.dataset import ICBHIDataset, get_dataloaders
from src.models import create_model
from src.metrics import MetricsCalculator, EarlyStopping
from src.features import FeatureExtractor
from src.visualize import plot_training_curves, plot_confusion_matrix


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int
) -> tuple:
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Train]")
    for batch_idx, (features, labels, metadata) in enumerate(pbar):
        features = features.to(device)
        labels = labels.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        logits, attention_weights = model(features)
        loss = criterion(logits, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Track metrics
        running_loss += loss.item()
        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        # Update progress bar
        pbar.set_postfix({'loss': loss.item()})
    
    avg_loss = running_loss / len(dataloader)
    accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
    
    return avg_loss, accuracy, np.array(all_labels), np.array(all_preds)


def validate_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    epoch: int
) -> tuple:
    """Validate for one epoch"""
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    all_probas = []
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Val]")
    with torch.no_grad():
        for features, labels, metadata in pbar:
            features = features.to(device)
            labels = labels.to(device)
            
            # Forward pass
            logits, attention_weights = model(features)
            loss = criterion(logits, labels)
            
            # Track metrics
            running_loss += loss.item()
            probas = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probas.extend(probas.cpu().numpy())
            
            # Update progress bar
            pbar.set_postfix({'loss': loss.item()})
    
    avg_loss = running_loss / len(dataloader)
    accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
    
    return avg_loss, accuracy, np.array(all_labels), np.array(all_preds), np.array(all_probas)


def save_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    epoch: int,
    metrics: dict,
    checkpoint_path: str,
    is_best: bool = False
):
    """Save model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics
    }
    
    torch.save(checkpoint, checkpoint_path)
    
    if is_best:
        best_path = checkpoint_path.replace('.pth', '_best.pth')
        torch.save(checkpoint, best_path)
        print(f"Saved best model to {best_path}")


def train(config: dict, args: argparse.Namespace):
    """Main training function"""
    # Set random seed
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    
    # Setup device
    device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create directories
    os.makedirs(config['paths']['checkpoint_dir'], exist_ok=True)
    os.makedirs(config['paths']['log_dir'], exist_ok=True)
    
    # Initialize feature extractor
    feature_config = config['features']
    feature_extractor = FeatureExtractor(**feature_config)
    
    # Create datasets
    print("Loading datasets...")
    train_dataset = ICBHIDataset(
        data_dir=config['paths']['data_dir'],
        split='train',
        feature_extractor=feature_extractor,
        augment=config['augmentation'],
        config=config
    )
    
    val_dataset = ICBHIDataset(
        data_dir=config['paths']['data_dir'],
        split='val',
        feature_extractor=feature_extractor,
        augment=False,
        config=config
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['training']['num_workers'],
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['training']['num_workers'],
        pin_memory=True
    )
    
    # Create model
    print("Creating model...")
    model = create_model(config, device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss function with class weights
    class_weights = train_dataset.class_weights.to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # Optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # Early stopping
    early_stopping = EarlyStopping(
        patience=config['training']['patience'],
        mode='max'  # Monitor validation ICBHI score
    )
    
    # Metrics calculator
    metrics_calc = MetricsCalculator()
    
    # Tensorboard writer
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = os.path.join(config['paths']['log_dir'], f'run_{timestamp}')
    writer = SummaryWriter(log_dir)
    
    # Training history
    history = {
        'train_loss': [],
        'train_accuracy': [],
        'val_loss': [],
        'val_accuracy': [],
        'val_icbhi_score': []
    }
    
    best_icbhi_score = 0.0
    
    print("\nStarting training...")
    print("=" * 80)
    
    for epoch in range(1, config['training']['num_epochs'] + 1):
        print(f"\nEpoch {epoch}/{config['training']['num_epochs']}")
        print("-" * 80)
        
        # Train
        train_loss, train_acc, train_labels, train_preds = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        
        # Validate
        val_loss, val_acc, val_labels, val_preds, val_probas = validate_epoch(
            model, val_loader, criterion, device, epoch
        )
        
        # Calculate detailed metrics
        val_metrics = metrics_calc.calculate_metrics(val_labels, val_preds, val_probas)
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Update history
        history['train_loss'].append(train_loss)
        history['train_accuracy'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_accuracy'].append(val_acc)
        history['val_icbhi_score'].append(val_metrics['icbhi_score'])
        
        # Log to tensorboard
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)
        writer.add_scalar('Accuracy/val', val_acc, epoch)
        writer.add_scalar('ICBHI_Score/val', val_metrics['icbhi_score'], epoch)
        writer.add_scalar('F1_Score/val', val_metrics['f1_macro'], epoch)
        
        # Print epoch summary
        print(f"\nTrain Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
        print(f"Val ICBHI Score: {val_metrics['icbhi_score']:.4f}")
        print(f"Val F1 (macro): {val_metrics['f1_macro']:.4f}")
        
        # Save checkpoint
        checkpoint_path = os.path.join(
            config['paths']['checkpoint_dir'],
            f'checkpoint_epoch_{epoch}.pth'
        )
        
        is_best = val_metrics['icbhi_score'] > best_icbhi_score
        if is_best:
            best_icbhi_score = val_metrics['icbhi_score']
            print(f"New best ICBHI score: {best_icbhi_score:.4f}")
        
        save_checkpoint(model, optimizer, epoch, val_metrics, checkpoint_path, is_best)
        
        # Early stopping
        if early_stopping(val_metrics['icbhi_score']):
            print(f"\nEarly stopping triggered after {epoch} epochs")
            break
    
    print("\n" + "=" * 80)
    print("Training completed!")
    print(f"Best ICBHI Score: {best_icbhi_score:.4f}")
    
    # Plot and save training curves
    curves_path = os.path.join(config['paths']['log_dir'], 'training_curves.png')
    plot_training_curves(history, metrics=['loss', 'accuracy'], save_path=curves_path)
    print(f"Training curves saved to {curves_path}")
    
    # Close tensorboard writer
    writer.close()


def main():
    parser = argparse.ArgumentParser(description='Train breath sound classification model')
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--data_dir', type=str, default=None,
                        help='Path to ICBHI dataset (overrides config)')
    parser.add_argument('--checkpoint_dir', type=str, default=None,
                        help='Path to save checkpoints (overrides config)')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume training')
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Override config with command line arguments
    if args.data_dir:
        config['paths']['data_dir'] = args.data_dir
    if args.checkpoint_dir:
        config['paths']['checkpoint_dir'] = args.checkpoint_dir
    
    # Train
    train(config, args)


if __name__ == '__main__':
    main()
