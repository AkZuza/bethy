"""
Feature extraction utilities for audio processing
"""

import torch
import torchaudio
import librosa
import numpy as np
from typing import Tuple, Optional


class FeatureExtractor:
    """Extract features from audio files"""
    
    def __init__(
        self,
        sample_rate: int = 16000,
        n_fft: int = 2048,
        hop_length: int = 512,
        n_mels: int = 128,
        n_mfcc: int = 40,
        duration: float = 8.0
    ):
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.n_mfcc = n_mfcc
        self.duration = duration
        self.target_length = int(sample_rate * duration)
        
        # Initialize transforms
        self.mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels
        )
        
        self.mfcc_transform = torchaudio.transforms.MFCC(
            sample_rate=sample_rate,
            n_mfcc=n_mfcc,
            melkwargs={
                'n_fft': n_fft,
                'hop_length': hop_length,
                'n_mels': n_mels
            }
        )
    
    def load_audio(self, audio_path: str) -> torch.Tensor:
        """Load and preprocess audio file"""
        waveform, sr = torchaudio.load(audio_path)
        
        # Resample if necessary
        if sr != self.sample_rate:
            resampler = torchaudio. transforms.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)
        
        # Convert to mono if stereo
        if waveform. shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # Pad or truncate to target length
        if waveform.shape[1] < self.target_length:
            pad_length = self.target_length - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, pad_length))
        else:
            waveform = waveform[:, :self.target_length]
        
        return waveform
    
    def extract_mel_spectrogram(self, waveform: torch.Tensor) -> torch.Tensor:
        """Extract mel-spectrogram"""
        mel_spec = self.mel_spectrogram(waveform)
        # Convert to log scale
        mel_spec = torch.log(mel_spec + 1e-9)
        return mel_spec
    
    def extract_mfcc(self, waveform:  torch.Tensor) -> torch.Tensor:
        """Extract MFCC features"""
        mfcc = self.mfcc_transform(waveform)
        return mfcc
    
    def extract_features(self, audio_path: str) -> dict:
        """Extract all features from audio file"""
        waveform = self.load_audio(audio_path)
        
        mel_spec = self.extract_mel_spectrogram(waveform)
        mfcc = self.extract_mfcc(waveform)
        
        return {
            'waveform': waveform,
            'mel_spectrogram': mel_spec,
            'mfcc': mfcc
        }
    
    def normalize(self, features: torch.Tensor) -> torch.Tensor:
        """Normalize features to zero mean and unit variance"""
        mean = features.mean()
        std = features.std()
        return (features - mean) / (std + 1e-9)


def compute_deltas(features: torch.Tensor, width: int = 9) -> torch.Tensor:
    """Compute delta features"""
    deltas = torch.zeros_like(features)
    for t in range(width, features.shape[-1] - width):
        deltas[..., t] = (features[..., t+1: t+width+1]. sum(dim=-1) - 
                          features[..., t-width:t].sum(dim=-1)) / (2 * width)
    return deltas