# Overfitting/Underfitting Fixes Documentation

## Problem Statement
Training and validation losses were both > 1, indicating potential overfitting or underfitting issues that needed investigation and resolution.

## Root Causes Identified

### 1. Feature Normalization Issues
**Problem**: Per-sample global normalization was inconsistent across batches, leading to unstable feature scales.

**Impact**: The model received features with varying scales, making it harder to learn consistent patterns.

**Fix**: Changed to frequency-axis normalization which:
- Preserves temporal patterns in the audio
- Ensures consistent feature scales across samples
- Maintains zero mean and unit variance along frequency dimension

### 2. Feature Scale Imbalance
**Problem**: Mel-spectrogram (128 dims) and MFCC (40 dims) were concatenated without considering their different scales.

**Impact**: The model could be biased toward features with larger magnitudes.

**Fix**: Applied consistent normalization to both feature types before concatenation.

### 3. Model Architecture Limitations
**Problem**: 
- No batch normalization in classifier layers
- LSTM input size was too large (5376 features)
- No proper weight initialization

**Impact**: 
- Training instability
- Inefficient feature representations
- Slower convergence

**Fix**: 
- Added batch normalization to classifier layers
- Added adaptive pooling to reduce feature dimensions from 21 to 4 in height
- Implemented proper weight initialization (Kaiming, Xavier, Orthogonal)

### 4. Training Instabilities
**Problem**: 
- No gradient clipping (risk of exploding gradients)
- No label smoothing (overfitting risk)
- No learning rate warmup
- Learning rate too low (0.0001)

**Impact**: 
- Potential gradient explosions
- Model overconfidence on training data
- Training instability in early epochs
- Slow convergence

**Fix**:
- Added gradient clipping (max_norm=1.0)
- Implemented label smoothing (0.1)
- Added 5-epoch learning rate warmup
- Increased learning rate to 0.0003

## Changes Made

### 1. `src/features.py`
```python
# Before
def normalize(self, features: torch.Tensor) -> torch.Tensor:
    mean = features.mean()
    std = features.std()
    return (features - mean) / (std + 1e-9)

# After
def normalize(self, features: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    # Normalize across frequency dimension (dim=1)
    mean = features.mean(dim=1, keepdim=True)
    std = features.std(dim=1, keepdim=True)
    return (features - mean) / (std + eps)
```

### 2. `src/models.py`
- Added `self.adaptive_pool = nn.AdaptiveAvgPool2d((4, None))` to reduce height dimension
- Added batch normalization layers to classifier
- Implemented `_init_weights()` method with proper initialization:
  - Kaiming initialization for Conv2d layers
  - Xavier initialization for Linear layers
  - Orthogonal initialization for LSTM hidden-to-hidden weights
  - Constant initialization for biases and BatchNorm

### 3. `train.py`
- Added gradient clipping: `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)`
- Added label smoothing: `nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)`
- Added warmup scheduler: `optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=5)`
- Added gradient norm tracking and logging
- Added learning rate logging to TensorBoard

### 4. `config.yaml`
- Increased learning rate: `0.0001` → `0.0003`

## Expected Impact

### Training Stability
- **Gradient Clipping**: Prevents exploding gradients, ensuring stable training
- **Weight Initialization**: Faster convergence and better gradient flow
- **Learning Rate Warmup**: Smoother training in early epochs

### Model Generalization
- **Label Smoothing**: Reduces overconfidence, improving generalization
- **Batch Normalization**: Better feature representations, faster convergence
- **Improved Normalization**: More consistent feature learning

### Loss Reduction
With these changes, you should expect:
- Lower training loss (< 1.0) within first 10-20 epochs
- Lower validation loss with smaller gap from training loss
- Better ICBHI scores (target: ≥ 75%)
- More stable loss curves

## Validation Tests

All changes have been validated with `test_improvements.py`:
- ✓ Feature normalization: Mean ~0, Std ~1 across frequency axis
- ✓ Model initialization: Proper weight distribution
- ✓ Gradient flow: Healthy gradient norms (not exploding/vanishing)
- ✓ Loss function: Label smoothing working correctly

## Monitoring Training

Key metrics to monitor:
1. **Train/Val Loss**: Should decrease steadily, both < 1.0
2. **Gradient Norm**: Should stay stable around 0.1-1.0
3. **Learning Rate**: Warmup for 5 epochs, then adaptive reduction
4. **ICBHI Score**: Should improve to ≥ 75%

Use TensorBoard to visualize:
```bash
tensorboard --logdir ./logs
```

## Next Steps

If losses are still high after these fixes:
1. **Check Data Quality**: Ensure audio files are properly preprocessed
2. **Verify Class Balance**: Review class weight computation
3. **Adjust Hyperparameters**: Try different learning rates (0.0001-0.001)
4. **Increase Model Capacity**: Consider deeper CNN or larger RNN hidden size
5. **Data Augmentation**: Review augmentation strength (may be too aggressive)

## References

- Kaiming He et al. "Delving Deep into Rectifiers" (weight initialization)
- Szegedy et al. "Rethinking the Inception Architecture" (label smoothing)
- Pascanu et al. "Understanding the exploding gradient problem" (gradient clipping)
- Ioffe & Szegedy "Batch Normalization: Accelerating Deep Network Training"
