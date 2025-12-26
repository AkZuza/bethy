# Bethy - Breath Sound Classification System

A deep learning-based system for classifying respiratory sounds into **wheezes**, **crackles**, or **both** using the ICBHI 2017 dataset. 

## Features

- 🎯 **Hybrid Architecture**: CNN-RNN with attention mechanism
- 📊 **Comprehensive Metrics**: F1 Score, Precision, Specificity, ICBHI Score
- 💾 **Model Checkpointing**:  Automatic saving of best models
- 📈 **Visualization**: Spectrograms, attention maps, training curves
- 🎚️ **Confidence Scores**:  Probability distribution for predictions
- ⚡ **Easy CLI**: Simple command-line interface for training and inference

## Installation

```bash
# Clone the repository
git clone https://github.com/akzuza/bethy.git
cd bethy

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Dataset Setup

1. Download the ICBHI 2017 Respiratory Sound Database
2. Extract to `data/icbhi/`
3. Structure should be:
```
data/icbhi/
├── audio/
│   ├── 101_1b1_Al_sc_Meditron. wav
│   ├── 101_1b1_Ar_sc_Meditron. wav
│   └── ...
└── ICBHI_final_database/
    ├── patient_diagnosis.csv
    └── respiratory_cycle_annotations.txt
```

## Usage

### Training

```bash
python train.py --data_dir ./data/icbhi --config config.yaml --checkpoint_dir ./checkpoints
```

### Single File Prediction

```bash
python predict.py --audio_path sample. wav --model_path ./checkpoints/best_model.pth --visualize
```

### Batch Prediction

```bash
python predict.py --batch_dir ./audio_samples/ --model_path ./checkpoints/best_model.pth --output results.csv
```

### Evaluation

```bash
python evaluate.py --data_dir ./data/icbhi --model_path ./checkpoints/best_model.pth --split test
```

## Model Architecture

Hybrid CNN-RNN-Attention architecture: 
- **CNN Backbone**: Extracts spatial features from spectrograms
- **Bidirectional LSTM**: Captures temporal patterns
- **Attention Mechanism**:  Focuses on important time-frequency regions
- **Multi-feature Input**: Supports mel-spectrograms, MFCCs

## Performance Target

- ICBHI Score:  ≥ 75%
- Balanced F1, Precision, and Specificity across classes

## Project Structure

```
bethy/
├── src/
│   ├── dataset.py          # ICBHI dataset loader
│   ├── models.py           # Hybrid CNN-RNN-Attention model
│   ├── features.py         # Feature extraction
│   ├── augmentation. py     # Data augmentation
│   ├── metrics.py          # Evaluation metrics
│   └── visualize.py        # Visualization utilities
├── train.py                # Training script
├── predict.py              # Inference script
├── evaluate. py             # Evaluation script
├── config.yaml             # Configuration
├── requirements.txt
└── README.md
```

## Citation

If you use this code, please cite the ICBHI dataset:

```
@article{rocha2019icbhi,
  title={An open access database for the evaluation of respiratory sound classification algorithms},
  author={Rocha, Bruno M and others},
  journal={Physiological measurement},
  volume={40},
  number={3},
  pages={035001},
  year={2019}
}
```

## License

MIT License