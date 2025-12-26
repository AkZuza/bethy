# Running Bethy on Kaggle

This guide will help you set up and run the Bethy breath sound classification project on Kaggle.

## Prerequisites

1. **Kaggle Account**: Sign up at [kaggle.com](https://www.kaggle.com)
2. **Dataset**: Add the ICBHI 2017 Respiratory Sound Database to your notebook
   - Search for "ICBHI Respiratory Sound Database" or "Respiratory Sound Database" on Kaggle Datasets
   - Common dataset links:
     - https://www.kaggle.com/datasets/vbookshelf/respiratory-sound-database
     - Or upload your own dataset

## Setup Methods

### Method 1: Using the Kaggle Notebook (Recommended)

1. **Upload Code to Kaggle**:
   - Create a new Kaggle notebook
   - Go to "File" → "Upload Notebook"
   - Upload `kaggle_notebook.ipynb`

2. **Add Code as Dataset** (Alternative):
   - Create a new dataset on Kaggle
   - Upload all `.py` files from the `src/` directory
   - Upload `config_kaggle.yaml`
   - Upload `train.py`, `evaluate.py`, `predict.py`

3. **Configure the Notebook**:
   - In the notebook, go to "Add data" → Search for ICBHI dataset
   - Add the respiratory sound database dataset
   - Enable GPU: Settings → Accelerator → GPU T4 x2 (or P100)
   - Enable Internet if needed for package installation

4. **Update Dataset Path**:
   - In the notebook, update the dataset path in cell 3 to match your dataset location
   - Example: `/kaggle/input/respiratory-sound-database/Respiratory_Sound_Database/Respiratory_Sound_Database`

5. **Run the Notebook**:
   - Run all cells sequentially
   - Training will take 2-4 hours depending on GPU and epochs

### Method 2: Using GitHub Integration

1. **Push Code to GitHub**:
   ```bash
   git add .
   git commit -m "Prepare for Kaggle"
   git push
   ```

2. **In Kaggle Notebook**:
   ```python
   # Clone repository
   !git clone https://github.com/AkZuza/bethy.git
   %cd bethy
   
   # Install dependencies
   !pip install -r requirements.txt
   ```

3. **Add Dataset and Run**:
   - Add ICBHI dataset through "Add data"
   - Update config_kaggle.yaml with correct paths
   - Run training:
   ```python
   !python train.py --config config_kaggle.yaml
   ```

## Dataset Path Configuration

The ICBHI dataset structure on Kaggle varies. Update `config_kaggle.yaml` based on your dataset:

### Common Structures:

**Option 1**: Direct structure
```yaml
paths:
  data_dir: "/kaggle/input/respiratory-sound-database"
```

**Option 2**: Nested structure
```yaml
paths:
  data_dir: "/kaggle/input/respiratory-sound-database/Respiratory_Sound_Database/Respiratory_Sound_Database"
```

**Option 3**: Custom upload
```yaml
paths:
  data_dir: "/kaggle/input/icbhi-2017"
```

### Verify Dataset Structure

Run this in a Kaggle notebook cell to check:
```python
import os
# List all input datasets
!ls -la /kaggle/input/

# Check specific dataset structure
!ls -la /kaggle/input/respiratory-sound-database/

# Find audio files
!find /kaggle/input -name "*.wav" | head -5
```

## Training Configuration for Kaggle

The `config_kaggle.yaml` is optimized for Kaggle with:
- **num_workers: 2** (Kaggle has 4 CPU cores)
- **batch_size: 32** (adjust based on GPU memory)
- **device: "cuda"** (use GPU)
- Output paths set to `/kaggle/working/`

### Adjust for Faster Training (Testing):
```yaml
training:
  batch_size: 64  # Increase if GPU has memory
  num_epochs: 10   # Reduce for quick testing
  patience: 5      # Early stopping
```

### Adjust for Better Performance:
```yaml
training:
  batch_size: 32
  num_epochs: 150
  patience: 20
  learning_rate: 0.00005  # Try lower learning rate
```

## Expected Output

After training, you'll find in `/kaggle/working/`:
- `checkpoints/` - Model checkpoints
- `logs/` - TensorBoard logs
- `bethy_final_model.pth` - Best trained model
- Training curves and visualizations

## Download Trained Model

After training completes:
1. Go to the "Output" section on the right side
2. Download `bethy_final_model.pth`
3. Download `config_final.yaml`

## Performance Tips

1. **GPU Usage**:
   - Enable GPU acceleration (Settings → GPU T4 x2)
   - Monitor GPU usage: `!nvidia-smi`

2. **Speed Up Training**:
   - Reduce `num_epochs` for testing
   - Increase `batch_size` if GPU memory allows
   - Set `num_workers: 2` (Kaggle limitation)

3. **Avoid Session Timeout**:
   - Save checkpoints frequently
   - Use early stopping
   - Kaggle notebooks timeout after 9-12 hours

4. **Memory Management**:
   - Kaggle provides ~30GB RAM
   - If OOM errors occur, reduce batch_size
   - Clear cache: `torch.cuda.empty_cache()`

## Common Issues and Solutions

### Issue 1: Dataset Not Found
```
FileNotFoundError: [Errno 2] No such file or directory
```
**Solution**: Verify dataset path. Check with `!ls /kaggle/input/`

### Issue 2: CUDA Out of Memory
```
RuntimeError: CUDA out of memory
```
**Solution**: Reduce batch_size in config:
```yaml
training:
  batch_size: 16  # or even 8
```

### Issue 3: Annotation File Not Found
```
FileNotFoundError: respiratory_cycle_annotations.txt
```
**Solution**: Update the annotation file path in `src/dataset.py` line 72 to match your dataset structure.

### Issue 4: Missing Dependencies
```
ModuleNotFoundError: No module named 'yaml'
```
**Solution**: Install in notebook:
```python
!pip install pyyaml tensorboard
```

## Monitoring Training

### View TensorBoard (if available):
```python
%load_ext tensorboard
%tensorboard --logdir /kaggle/working/logs
```

### Check Training Progress:
The training script prints:
- Loss and accuracy per epoch
- Validation metrics
- ICBHI Score
- Best model notifications

## Evaluation After Training

```python
# In notebook
from evaluate import main as evaluate_main
import sys

sys.argv = [
    'evaluate.py',
    '--model_path', '/kaggle/working/checkpoints/checkpoint_epoch_50_best.pth',
    '--split', 'test',
    '--visualize'
]

evaluate_main()
```

## Making Predictions

```python
from predict import BreathSoundPredictor
import yaml

# Load config
with open('config_kaggle.yaml') as f:
    config = yaml.safe_load(f)

# Initialize predictor
predictor = BreathSoundPredictor(
    model_path='/kaggle/working/checkpoints/checkpoint_epoch_50_best.pth',
    config=config,
    device=torch.device('cuda')
)

# Predict
result = predictor.predict_single('/kaggle/input/your-audio.wav')
print(f"Predicted: {result['predicted_class']}")
print(f"Confidence: {result['confidence']:.2%}")
```

## Resources

- [Kaggle Notebooks Documentation](https://www.kaggle.com/docs/notebooks)
- [ICBHI 2017 Challenge](https://bhichallenge.med.auth.gr/)
- [Project Repository](https://github.com/AkZuza/bethy)

## Support

If you encounter issues:
1. Check the "Common Issues" section above
2. Verify dataset paths and structure
3. Check Kaggle notebook logs for detailed errors
4. Review the main README.md for general project info
