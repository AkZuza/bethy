"""
Data augmentation techniques for audio
"""

import torch
import torchaudio
import numpy as np
import random
from typing import Tuple


class AudioAugmentation:
    """Audio augmentation techniques"""
    
    def __init__(
        self,
        sample_rate: int = 16000,
        time_stretch_rate:  Tuple[float, float] = (0.8, 1.2),
        pitch_shift_steps: int = 4,
        noise_factor: float = 0.005
    ):
        self.sample_rate = sample_rate
        self.time_stretch_rate = time_stretch_rate
        self. pitch_shift_steps = pitch_shift_steps
        self.noise_factor = noise_factor
    
    def time_stretch(self, waveform: torch.Tensor) -> torch.Tensor:
        """Apply time stretching"""
        rate = random.uniform(*self.time_stretch_rate)
        # Use torchaudio's speed perturbation
        effects = [["tempo", str(rate)]]
        waveform_stretched, _ = torchaudio.sox_effects.apply_effects_tensor(
            waveform, self.sample_rate, effects
        )
        return waveform_stretched
    
    def pitch_shift(self, waveform:  torch.Tensor) -> torch.Tensor:
        """Apply pitch shifting"""
        n_steps = random.randint(-self.pitch_shift_steps, self.pitch_shift_steps)
        effects = [["pitch", str(n_steps * 100)]]
        waveform_shifted, _ = torchaudio.sox_effects.apply_effects_tensor(
            waveform, self.sample_rate, effects
        )
        return waveform_shifted
    
    def add_noise(self, waveform: torch.Tensor) -> torch.Tensor:
        """Add Gaussian noise"""
        noise = torch.randn_like(waveform) * self.noise_factor
        return waveform + noise
    
    def random_gain(self, waveform: torch.Tensor, min_gain: float = 0.5, max_gain: float = 1.5) -> torch.Tensor:
        """Apply random gain"""
        gain = random.uniform(min_gain, max_gain)
        return waveform * gain
    
    def apply_random_augmentation(self, waveform:  torch.Tensor) -> torch.Tensor:
        """Apply random combination of augmentations"""
        augmentations = []
        
        if random.random() > 0.5:
            augmentations.append(self.add_noise)
        
        if random.random() > 0.5:
            augmentations. append(self.random_gain)
        
        # Apply selected augmentations
        for aug in augmentations:
            try:
                waveform = aug(waveform)
            except: 
                continue
        
        return waveform


class SpecAugment:
    """SpecAugment for spectrograms"""
    
    def __init__(
        self,
        freq_mask_param: int = 30,
        time_mask_param: int = 40,
        num_masks: int = 2
    ):
        self.freq_masking = torchaudio.transforms.FrequencyMasking(freq_mask_param)
        self.time_masking = torchaudio.transforms.TimeMasking(time_mask_param)
        self.num_masks = num_masks
    
    def __call__(self, spectrogram: torch.Tensor) -> torch.Tensor:
        """Apply SpecAugment"""
        for _ in range(self.num_masks):
            spectrogram = self.freq_masking(spectrogram)
            spectrogram = self.time_masking(spectrogram)
        return spectrogram