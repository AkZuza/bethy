"""
ICBHI Dataset loader and preprocessing
"""

import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from typing import List, Dict, Tuple, Optional
import numpy as np
from src.features import FeatureExtractor
from src.augmentation import AudioAugmentation, SpecAugment


class ICBHIDataset(Dataset):
    """ICBHI Respiratory Sound Database Dataset"""
    
    # Class mapping
    CLASS_MAPPING = {
        (0, 0): 0,  # normal
        (1, 0): 1,  # wheeze
        (0, 1): 2,  # crackle
        (1, 1): 3   # both
    }
    
    CLASS_NAMES = ['normal', 'wheeze', 'crackle', 'both']
    
    def __init__(
        self,
        data_dir:  str,
        split: str = 'train',
        feature_extractor: Optional[FeatureExtractor] = None,
        augment:  bool = False,
        config: Optional[dict] = None
    ):
        """
        Args:
            data_dir: Path to ICBHI dataset
            split: 'train', 'val', or 'test'
            feature_extractor: Feature extraction object
            augment: Whether to apply data augmentation
            config: Configuration dictionary
        """
        self.data_dir = data_dir
        self.split = split
        self.augment = augment and split == 'train'
        
        # Initialize feature extractor
        if feature_extractor is None: 
            self.feature_extractor = FeatureExtractor()
        else:
            self. feature_extractor = feature_extractor
        
        # Initialize augmentation
        if self.augment:
            self. audio_augment = AudioAugmentation()
            self.spec_augment = SpecAugment()
        
        # Load annotations
        self.samples = self._load_annotations()
        
        # Compute class weights for imbalanced dataset
        self.class_weights = self._compute_class_weights()
    
    def _load_annotations(self) -> List[Dict]:
        """Load ICBHI annotations"""
        annotation_file = os.path.join(
            self.data_dir, 
            'ICBHI_final_database', 
            'respiratory_cycle_annotations.txt'
        )
        
        samples = []
        
        # Read annotation file
        with open(annotation_file, 'r') as f:
            for line in f: 
                parts = line.strip().split('\t')
                if len(parts) < 4:
                    continue
                
                filename = parts[0]
                start_time = float(parts[1])
                end_time = float(parts[2])
                crackles = int(parts[3])
                wheezes = int(parts[4]) if len(parts) > 4 else 0
                
                # Get full audio path
                audio_path = os.path.join(self.data_dir, 'audio', filename + '.wav')
                
                if not os.path.exists(audio_path):
                    continue
                
                # Map to class label
                label = self.CLASS_MAPPING[(wheezes, crackles)]
                
                samples.append({
                    'audio_path': audio_path,
                    'filename': filename,
                    'start_time': start_time,
                    'end_time': end_time,
                    'crackles': crackles,
                    'wheezes': wheezes,
                    'label': label
                })
        
        # Split data (using patient-wise split for ICBHI protocol)
        samples = self._split_data(samples)
        
        return samples
    
    def _split_data(self, samples: List[Dict]) -> List[Dict]:
        """Split data following ICBHI protocol (patient-wise)"""
        # Extract patient IDs from filenames
        patient_ids = list(set([s['filename'].split('_')[0] for s in samples]))
        patient_ids.sort()
        
        # 60% train, 20% val, 20% test
        n_train = int(0.6 * len(patient_ids))
        n_val = int(0.2 * len(patient_ids))
        
        train_patients = set(patient_ids[:n_train])
        val_patients = set(patient_ids[n_train:n_train + n_val])
        test_patients = set(patient_ids[n_train + n_val:])
        
        # Filter samples based on split
        if self.split == 'train': 
            return [s for s in samples if s['filename'].split('_')[0] in train_patients]
        elif self.split == 'val':
            return [s for s in samples if s['filename'].split('_')[0] in val_patients]
        else:  # test
            return [s for s in samples if s['filename'].split('_')[0] in test_patients]
    
    def _compute_class_weights(self) -> torch.Tensor:
        """Compute class weights for imbalanced dataset"""
        labels = [s['label'] for s in self.samples]
        class_counts = np.bincount(labels, minlength=4)
        total = len(labels)
        weights = total / (len(class_counts) * class_counts + 1e-6)
        return torch.FloatTensor(weights)
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, Dict]:
        """Get a sample"""
        sample = self.samples[idx]
        
        # Extract features
        features = self.feature_extractor. extract_features(sample['audio_path'])
        
        # Apply augmentation if training
        if self.augment:
            features['waveform'] = self.audio_augment. apply_random_augmentation(
                features['waveform']
            )
            # Recompute spectrogram after augmentation
            features['mel_spectrogram'] = self.feature_extractor.extract_mel_spectrogram(
                features['waveform']
            )
            features['mel_spectrogram'] = self. spec_augment(features['mel_spectrogram'])
        
        # Normalize features
        mel_spec = self.feature_extractor.normalize(features['mel_spectrogram'])
        mfcc = self.feature_extractor.normalize(features['mfcc'])
        
        # Combine features - ensure proper channel dimension for CNN input
        # mel_spec shape: (1, n_mels, time), mfcc shape: (1, n_mfcc, time)
        # Concatenate along frequency dimension to get (1, n_mels+n_mfcc, time)
        combined_features = torch.cat([mel_spec, mfcc], dim=1)
        
        # Add channel dimension if needed for CNN: (1, n_mels+n_mfcc, time) -> keep as is
        # The model expects (batch, channels, height, width) format
        
        label = sample['label']
        
        metadata = {
            'filename': sample['filename'],
            'audio_path': sample['audio_path'],
            'crackles': sample['crackles'],
            'wheezes':  sample['wheezes']
        }
        
        return combined_features, label, metadata


def get_dataloaders(config: dict) -> Tuple[torch.utils.data.DataLoader, ... ]:
    """Create train, validation, and test dataloaders"""
    
    # Initialize feature extractor
    feature_config = config['features']
    feature_extractor = FeatureExtractor(**feature_config)
    
    # Create datasets
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
    
    test_dataset = ICBHIDataset(
        data_dir=config['paths']['data_dir'],
        split='test',
        feature_extractor=feature_extractor,
        augment=False,
        config=config
    )
    
    # Create dataloaders
    train_loader = torch. utils.data.DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['training']['num_workers'],
        pin_memory=True
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['training']['num_workers'],
        pin_memory=True
    )
    
    test_loader = torch. utils.data.DataLoader(
        test_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['training']['num_workers'],
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader, train_dataset. class_weights