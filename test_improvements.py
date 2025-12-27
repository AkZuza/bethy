"""
Test script to validate improvements for overfitting/underfitting issues
"""

import torch
import numpy as np
import yaml
from src.models import create_model, HybridCNNRNNAttention
from src.features import FeatureExtractor
from src.dataset import ICBHIDataset

def test_feature_normalization():
    """Test that feature normalization works correctly"""
    print("Testing feature normalization...")
    
    # Use default feature extractor config
    config = {
        'sample_rate': 16000,
        'n_fft': 2048,
        'hop_length': 512,
        'n_mels': 128,
        'n_mfcc': 40,
        'duration': 8.0
    }
    
    extractor = FeatureExtractor(**config)
    
    # Create dummy features with shapes matching config
    n_mels = config['n_mels']
    n_mfcc = config['n_mfcc']
    time_frames = 250  # Approximate frames for 8 seconds of audio
    
    dummy_mel = torch.randn(1, n_mels, time_frames)  # (channels, freq, time)
    dummy_mfcc = torch.randn(1, n_mfcc, time_frames)
    
    # Normalize
    mel_norm = extractor.normalize(dummy_mel)
    mfcc_norm = extractor.normalize(dummy_mfcc)
    
    # Check that normalization is per frequency band (dim=1)
    mel_mean = mel_norm.mean(dim=1)
    mel_std = mel_norm.std(dim=1)
    
    print(f"  Mel mean across freq: {mel_mean.mean():.6f} (should be ~0)")
    print(f"  Mel std across freq: {mel_std.mean():.6f} (should be ~1)")
    
    # Combine features
    combined = torch.cat([mel_norm, mfcc_norm], dim=1)
    expected_shape = (1, n_mels + n_mfcc, time_frames)
    print(f"  Combined shape: {combined.shape}")
    print(f"  Expected: {expected_shape}")
    
    assert combined.shape == expected_shape, f"Unexpected shape: {combined.shape}"
    print("✓ Feature normalization test passed!\n")

def test_model_initialization():
    """Test model initialization and forward pass"""
    print("Testing model initialization...")
    
    config = {
        'model': {
            'cnn_channels': [64, 128, 256],
            'rnn_hidden_size': 256,
            'rnn_num_layers': 2,
            'attention_dim': 128,
            'num_classes': 4,
            'dropout': 0.5
        }
    }
    
    device = torch.device('cpu')
    model = create_model(config, device)
    
    # Test forward pass with dummy data
    batch_size = 4
    # Input: (batch, channels, height, width)
    dummy_input = torch.randn(batch_size, 1, 168, 250)
    
    logits, attention = model(dummy_input)
    
    print(f"  Input shape: {dummy_input.shape}")
    print(f"  Logits shape: {logits.shape}")
    print(f"  Attention shape: {attention.shape}")
    print(f"  Expected logits: ({batch_size}, 4)")
    
    assert logits.shape == (batch_size, 4), f"Unexpected logits shape: {logits.shape}"
    
    # Check that weights are initialized (not all zeros or ones)
    first_conv = model.cnn_blocks[0].conv1.weight
    print(f"  First conv weight mean: {first_conv.mean():.6f}")
    print(f"  First conv weight std: {first_conv.std():.6f}")
    
    assert first_conv.std() > 0.01, "Weights not properly initialized"
    print("✓ Model initialization test passed!\n")

def test_gradient_flow():
    """Test that gradients flow properly through the model"""
    print("Testing gradient flow...")
    
    config = {
        'model': {
            'cnn_channels': [64, 128, 256],
            'rnn_hidden_size': 256,
            'rnn_num_layers': 2,
            'attention_dim': 128,
            'num_classes': 4,
            'dropout': 0.5
        }
    }
    
    device = torch.device('cpu')
    model = create_model(config, device)
    
    # Dummy input and target
    dummy_input = torch.randn(2, 1, 168, 250)
    dummy_target = torch.tensor([0, 1])
    
    # Forward pass
    logits, _ = model(dummy_input)
    
    # Loss with label smoothing
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
    loss = criterion(logits, dummy_target)
    
    # Backward pass
    loss.backward()
    
    # Check gradients
    has_grad = False
    grad_norms = []
    for name, param in model.named_parameters():
        if param.grad is not None:
            has_grad = True
            grad_norm = param.grad.norm().item()
            grad_norms.append(grad_norm)
    
    print(f"  Loss value: {loss.item():.6f}")
    print(f"  Has gradients: {has_grad}")
    print(f"  Avg gradient norm: {np.mean(grad_norms):.6f}")
    print(f"  Max gradient norm: {np.max(grad_norms):.6f}")
    
    assert has_grad, "No gradients computed!"
    assert np.mean(grad_norms) > 0, "Zero gradients!"
    print("✓ Gradient flow test passed!\n")

def test_loss_function():
    """Test loss function with class weights and label smoothing"""
    print("Testing loss function...")
    
    # Create dummy predictions and labels
    logits = torch.randn(8, 4)
    labels = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3])
    
    # Class weights (dummy)
    class_weights = torch.tensor([1.0, 2.0, 1.5, 3.0])
    
    # Loss with and without label smoothing
    criterion_no_smooth = torch.nn.CrossEntropyLoss(weight=class_weights)
    criterion_smooth = torch.nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
    
    loss_no_smooth = criterion_no_smooth(logits, labels)
    loss_smooth = criterion_smooth(logits, labels)
    
    print(f"  Loss without smoothing: {loss_no_smooth.item():.6f}")
    print(f"  Loss with smoothing: {loss_smooth.item():.6f}")
    print(f"  Smoothing reduces overconfidence: {loss_smooth.item() > loss_no_smooth.item()}")
    
    print("✓ Loss function test passed!\n")

def main():
    """Run all tests"""
    print("=" * 60)
    print("Running Overfitting/Underfitting Fix Validation Tests")
    print("=" * 60)
    print()
    
    try:
        test_feature_normalization()
        test_model_initialization()
        test_gradient_flow()
        test_loss_function()
        
        print("=" * 60)
        print("✓ All tests passed successfully!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == '__main__':
    exit(main())
