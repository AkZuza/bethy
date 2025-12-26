"""
Prediction script for breath sound classification
"""

import os
import argparse
import yaml
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

from src.models import create_model
from src.features import FeatureExtractor
from src.visualize import (
    plot_spectrogram,
    plot_attention_weights,
    plot_attention_on_spectrogram,
    plot_prediction_confidence
)


class BreathSoundPredictor:
    """Predictor for breath sound classification"""
    
    def __init__(
        self,
        model_path: str,
        config: dict,
        device: torch.device
    ):
        self.config = config
        self.device = device
        self.class_names = ['normal', 'wheeze', 'crackle', 'both']
        
        # Initialize feature extractor
        self.feature_extractor = FeatureExtractor(**config['features'])
        
        # Load model
        self.model = create_model(config, device)
        checkpoint = torch.load(model_path, map_location=device)
        
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
        
        self.model.eval()
    
    def predict_single(
        self,
        audio_path: str,
        return_attention: bool = True
    ) -> dict:
        """
        Predict class for a single audio file
        
        Args:
            audio_path: Path to audio file
            return_attention: Whether to return attention weights
        
        Returns:
            Dictionary with prediction results
        """
        # Extract features
        features_dict = self.feature_extractor.extract_features(audio_path)
        
        # Normalize features
        mel_spec = self.feature_extractor.normalize(features_dict['mel_spectrogram'])
        mfcc = self.feature_extractor.normalize(features_dict['mfcc'])
        
        # Combine features
        combined_features = torch.cat([mel_spec, mfcc], dim=1)
        
        # Add batch dimension
        features_tensor = combined_features.unsqueeze(0).to(self.device)
        
        # Predict
        with torch.no_grad():
            logits, attention_weights = self.model(features_tensor)
            probabilities = torch.softmax(logits, dim=1)
        
        # Get prediction
        pred_idx = torch.argmax(probabilities, dim=1).item()
        pred_class = self.class_names[pred_idx]
        confidence = probabilities[0, pred_idx].item()
        
        # Prepare results
        results = {
            'predicted_class': pred_class,
            'predicted_index': pred_idx,
            'confidence': confidence,
            'probabilities': {
                name: prob.item() 
                for name, prob in zip(self.class_names, probabilities[0])
            },
            'mel_spectrogram': mel_spec[0].cpu().numpy(),
            'mfcc': mfcc[0].cpu().numpy()
        }
        
        if return_attention:
            results['attention_weights'] = attention_weights[0].cpu().numpy()
        
        return results
    
    def predict_batch(
        self,
        audio_paths: list,
        batch_size: int = 8
    ) -> list:
        """
        Predict classes for multiple audio files
        
        Args:
            audio_paths: List of paths to audio files
            batch_size: Batch size for prediction
        
        Returns:
            List of prediction results
        """
        all_results = []
        
        for i in tqdm(range(0, len(audio_paths), batch_size), desc="Predicting"):
            batch_paths = audio_paths[i:i + batch_size]
            batch_features = []
            
            # Extract features for batch
            for audio_path in batch_paths:
                try:
                    features_dict = self.feature_extractor.extract_features(audio_path)
                    mel_spec = self.feature_extractor.normalize(features_dict['mel_spectrogram'])
                    mfcc = self.feature_extractor.normalize(features_dict['mfcc'])
                    combined = torch.cat([mel_spec, mfcc], dim=1)
                    batch_features.append(combined)
                except Exception as e:
                    print(f"Error processing {audio_path}: {e}")
                    all_results.append({
                        'audio_path': audio_path,
                        'error': str(e),
                        'predicted_class': None,
                        'confidence': None
                    })
                    continue
            
            if not batch_features:
                continue
            
            # Stack into batch tensor
            features_tensor = torch.stack(batch_features).to(self.device)
            
            # Predict
            with torch.no_grad():
                logits, attention_weights = self.model(features_tensor)
                probabilities = torch.softmax(logits, dim=1)
                predictions = torch.argmax(probabilities, dim=1)
            
            # Process results
            for j, audio_path in enumerate(batch_paths[:len(batch_features)]):
                pred_idx = predictions[j].item()
                pred_class = self.class_names[pred_idx]
                confidence = probabilities[j, pred_idx].item()
                
                results = {
                    'audio_path': audio_path,
                    'predicted_class': pred_class,
                    'predicted_index': pred_idx,
                    'confidence': confidence,
                    'probabilities': {
                        name: prob.item()
                        for name, prob in zip(self.class_names, probabilities[j])
                    }
                }
                all_results.append(results)
        
        return all_results


def main():
    parser = argparse.ArgumentParser(description='Predict breath sound classification')
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to trained model checkpoint')
    parser.add_argument('--audio_path', type=str, default=None,
                        help='Path to single audio file')
    parser.add_argument('--batch_dir', type=str, default=None,
                        help='Directory containing audio files for batch prediction')
    parser.add_argument('--output', type=str, default=None,
                        help='Output path for results (CSV for batch, optional)')
    parser.add_argument('--visualize', action='store_true',
                        help='Generate visualization plots')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for batch prediction')
    args = parser.parse_args()
    
    # Validate input
    if args.audio_path is None and args.batch_dir is None:
        parser.error("Either --audio_path or --batch_dir must be specified")
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Setup device
    device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize predictor
    print("Loading model...")
    predictor = BreathSoundPredictor(args.model_path, config, device)
    print("Model loaded successfully!")
    
    # Single file prediction
    if args.audio_path:
        print(f"\nPredicting for: {args.audio_path}")
        
        if not os.path.exists(args.audio_path):
            print(f"Error: Audio file not found: {args.audio_path}")
            return
        
        results = predictor.predict_single(args.audio_path)
        
        # Print results
        print("\n" + "=" * 60)
        print("Prediction Results")
        print("=" * 60)
        print(f"Predicted Class: {results['predicted_class'].upper()}")
        print(f"Confidence: {results['confidence']:.2%}")
        print("\nClass Probabilities:")
        for class_name, prob in results['probabilities'].items():
            print(f"  {class_name.capitalize():<10}: {prob:.2%}")
        print("=" * 60)
        
        # Visualize if requested
        if args.visualize:
            output_dir = os.path.dirname(args.audio_path) if args.output is None else os.path.dirname(args.output)
            if not output_dir:
                output_dir = '.'
            
            base_name = os.path.splitext(os.path.basename(args.audio_path))[0]
            
            # Plot spectrogram
            spec_path = os.path.join(output_dir, f'{base_name}_spectrogram.png')
            plot_spectrogram(
                results['mel_spectrogram'],
                title=f'Mel-Spectrogram: {results["predicted_class"].upper()} ({results["confidence"]:.2%})',
                save_path=spec_path
            )
            print(f"\nSpectrogram saved to {spec_path}")
            
            # Plot attention weights
            if 'attention_weights' in results:
                attn_path = os.path.join(output_dir, f'{base_name}_attention.png')
                plot_attention_weights(
                    results['attention_weights'],
                    title=f'Attention Weights: {results["predicted_class"].upper()}',
                    save_path=attn_path
                )
                print(f"Attention weights saved to {attn_path}")
                
                # Plot attention on spectrogram
                attn_spec_path = os.path.join(output_dir, f'{base_name}_attention_spectrogram.png')
                plot_attention_on_spectrogram(
                    results['mel_spectrogram'],
                    results['attention_weights'],
                    title=f'Attention on Spectrogram: {results["predicted_class"].upper()}',
                    save_path=attn_spec_path
                )
                print(f"Attention on spectrogram saved to {attn_spec_path}")
            
            # Plot prediction confidence
            conf_path = os.path.join(output_dir, f'{base_name}_confidence.png')
            probas = np.array([results['probabilities'][name] for name in predictor.class_names])
            plot_prediction_confidence(
                probas,
                predictor.class_names,
                predicted_label=results['predicted_index'],
                title=f'Prediction Confidence',
                save_path=conf_path
            )
            print(f"Confidence plot saved to {conf_path}")
    
    # Batch prediction
    elif args.batch_dir:
        print(f"\nProcessing audio files from: {args.batch_dir}")
        
        if not os.path.exists(args.batch_dir):
            print(f"Error: Directory not found: {args.batch_dir}")
            return
        
        # Find all audio files
        audio_extensions = ['.wav', '.mp3', '.flac', '.ogg']
        audio_paths = []
        
        for ext in audio_extensions:
            audio_paths.extend(Path(args.batch_dir).glob(f'*{ext}'))
            audio_paths.extend(Path(args.batch_dir).glob(f'**/*{ext}'))
        
        audio_paths = sorted(list(set([str(p) for p in audio_paths])))
        
        if not audio_paths:
            print(f"No audio files found in {args.batch_dir}")
            return
        
        print(f"Found {len(audio_paths)} audio files")
        
        # Predict
        results = predictor.predict_batch(audio_paths, batch_size=args.batch_size)
        
        # Create DataFrame
        df = pd.DataFrame([
            {
                'filename': os.path.basename(r['audio_path']),
                'predicted_class': r['predicted_class'],
                'confidence': r.get('confidence', None),
                **{f'prob_{name}': r.get('probabilities', {}).get(name, None) 
                   for name in predictor.class_names}
            }
            for r in results
        ])
        
        # Save to CSV
        output_path = args.output if args.output else os.path.join(args.batch_dir, 'predictions.csv')
        df.to_csv(output_path, index=False)
        print(f"\nPredictions saved to {output_path}")
        
        # Print summary
        print("\n" + "=" * 60)
        print("Batch Prediction Summary")
        print("=" * 60)
        print(f"Total files: {len(results)}")
        print("\nClass distribution:")
        class_counts = df['predicted_class'].value_counts()
        for class_name, count in class_counts.items():
            print(f"  {class_name.capitalize():<10}: {count} ({count/len(results)*100:.1f}%)")
        print(f"\nAverage confidence: {df['confidence'].mean():.2%}")
        print("=" * 60)


if __name__ == '__main__':
    main()
