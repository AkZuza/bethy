"""
Evaluation script for breath sound classification model
"""

import os
import argparse
import yaml
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset import ICBHIDataset
from src.models import create_model
from src.metrics import MetricsCalculator
from src.features import FeatureExtractor
from src.visualize import (
    plot_confusion_matrix,
    plot_class_distribution,
    save_visualization_grid
)


def evaluate_model(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    metrics_calc: MetricsCalculator
) -> dict:
    """Evaluate model on dataset"""
    model.eval()
    
    all_preds = []
    all_labels = []
    all_probas = []
    all_attention_weights = []
    all_spectrograms = []
    
    print("Evaluating model...")
    with torch.no_grad():
        for features, labels, metadata in tqdm(dataloader):
            features = features.to(device)
            
            # Forward pass
            logits, attention_weights = model(features)
            probas = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)
            
            # Collect results
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probas.extend(probas.cpu().numpy())
            all_attention_weights.extend(attention_weights.cpu().numpy())
            
            # Store first channel of features for visualization (mel-spectrogram)
            all_spectrograms.extend(features[:, 0, :, :].cpu().numpy())
    
    # Convert to numpy arrays
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probas = np.array(all_probas)
    
    # Calculate metrics
    metrics = metrics_calc.calculate_metrics(all_labels, all_preds, all_probas)
    
    # Add classification report
    report = metrics_calc.get_classification_report(all_labels, all_preds)
    
    results = {
        'metrics': metrics,
        'report': report,
        'labels': all_labels,
        'predictions': all_preds,
        'probabilities': all_probas,
        'attention_weights': all_attention_weights,
        'spectrograms': all_spectrograms
    }
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Evaluate breath sound classification model')
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to trained model checkpoint')
    parser.add_argument('--data_dir', type=str, default=None,
                        help='Path to ICBHI dataset (overrides config)')
    parser.add_argument('--split', type=str, default='test',
                        choices=['train', 'val', 'test'],
                        help='Dataset split to evaluate on')
    parser.add_argument('--output_dir', type=str, default='./evaluation_results',
                        help='Directory to save evaluation results')
    parser.add_argument('--visualize', action='store_true',
                        help='Generate visualization plots')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for evaluation')
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Override config with command line arguments
    if args.data_dir:
        config['paths']['data_dir'] = args.data_dir
    
    # Setup device
    device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize feature extractor
    feature_config = config['features']
    feature_extractor = FeatureExtractor(**feature_config)
    
    # Create dataset
    print(f"Loading {args.split} dataset...")
    dataset = ICBHIDataset(
        data_dir=config['paths']['data_dir'],
        split=args.split,
        feature_extractor=feature_extractor,
        augment=False,
        config=config
    )
    print(f"Dataset size: {len(dataset)}")
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=config['training']['num_workers'],
        pin_memory=True
    )
    
    # Create model
    print("Loading model...")
    model = create_model(config, device)
    
    # Load checkpoint
    checkpoint = torch.load(args.model_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded model from epoch {checkpoint.get('epoch', 'unknown')}")
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    
    # Initialize metrics calculator
    metrics_calc = MetricsCalculator()
    
    # Evaluate
    results = evaluate_model(model, dataloader, device, metrics_calc)
    
    # Print metrics
    print("\n" + "=" * 80)
    print(f"Evaluation Results on {args.split.upper()} set")
    print("=" * 80)
    metrics_calc.print_metrics(results['metrics'])
    
    print("\nClassification Report:")
    print(results['report'])
    
    # Save metrics to file
    metrics_path = os.path.join(args.output_dir, f'{args.split}_metrics.txt')
    with open(metrics_path, 'w') as f:
        f.write(f"Evaluation Results on {args.split.upper()} set\n")
        f.write("=" * 80 + "\n")
        f.write(f"Accuracy: {results['metrics']['accuracy']:.4f}\n")
        f.write(f"ICBHI Score: {results['metrics']['icbhi_score']:.4f}\n")
        f.write(f"F1 (macro): {results['metrics']['f1_macro']:.4f}\n")
        f.write(f"Precision (macro): {results['metrics']['precision_macro']:.4f}\n")
        f.write(f"Recall (macro): {results['metrics']['recall_macro']:.4f}\n")
        f.write(f"Specificity (macro): {results['metrics']['specificity_macro']:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(results['report'])
    
    print(f"\nMetrics saved to {metrics_path}")
    
    # Generate visualizations
    if args.visualize:
        print("\nGenerating visualizations...")
        
        # Confusion matrix
        cm_path = os.path.join(args.output_dir, f'{args.split}_confusion_matrix.png')
        plot_confusion_matrix(
            results['metrics']['confusion_matrix'],
            dataset.CLASS_NAMES,
            title=f'Confusion Matrix ({args.split.upper()} set)',
            save_path=cm_path
        )
        print(f"Confusion matrix saved to {cm_path}")
        
        # Normalized confusion matrix
        cm_norm_path = os.path.join(args.output_dir, f'{args.split}_confusion_matrix_normalized.png')
        plot_confusion_matrix(
            results['metrics']['confusion_matrix'],
            dataset.CLASS_NAMES,
            title=f'Confusion Matrix - Normalized ({args.split.upper()} set)',
            save_path=cm_norm_path,
            normalize=True
        )
        print(f"Normalized confusion matrix saved to {cm_norm_path}")
        
        # Class distribution
        dist_path = os.path.join(args.output_dir, f'{args.split}_class_distribution.png')
        plot_class_distribution(
            results['labels'],
            dataset.CLASS_NAMES,
            title=f'Class Distribution ({args.split.upper()} set)',
            save_path=dist_path
        )
        print(f"Class distribution saved to {dist_path}")
        
        # Sample visualizations with attention
        grid_path = os.path.join(args.output_dir, f'{args.split}_sample_predictions.png')
        save_visualization_grid(
            results['spectrograms'][:16],
            results['attention_weights'][:16],
            results['labels'][:16].tolist(),
            results['predictions'][:16].tolist(),
            dataset.CLASS_NAMES,
            save_path=grid_path,
            max_samples=16
        )
        print(f"Sample predictions grid saved to {grid_path}")
    
    # Save predictions
    predictions_path = os.path.join(args.output_dir, f'{args.split}_predictions.npz')
    np.savez(
        predictions_path,
        labels=results['labels'],
        predictions=results['predictions'],
        probabilities=results['probabilities']
    )
    print(f"\nPredictions saved to {predictions_path}")
    
    print("\n" + "=" * 80)
    print("Evaluation completed!")
    print("=" * 80)


if __name__ == '__main__':
    main()
