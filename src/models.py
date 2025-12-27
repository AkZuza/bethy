"""
Hybrid CNN-RNN-Attention model for breath sound classification
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class AttentionLayer(nn.Module):
    """Attention mechanism for focusing on important features"""
    
    def __init__(self, hidden_dim: int, attention_dim: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, attention_dim),
            nn.Tanh(),
            nn.Linear(attention_dim, 1)
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch, seq_len, hidden_dim)
        Returns:
            context: (batch, hidden_dim)
            attention_weights: (batch, seq_len)
        """
        # Compute attention scores
        attention_scores = self. attention(x).squeeze(-1)  # (batch, seq_len)
        attention_weights = F.softmax(attention_scores, dim=1)  # (batch, seq_len)
        
        # Apply attention weights
        context = torch.bmm(
            attention_weights.unsqueeze(1),  # (batch, 1, seq_len)
            x  # (batch, seq_len, hidden_dim)
        ).squeeze(1)  # (batch, hidden_dim)
        
        return context, attention_weights


class CNNBlock(nn.Module):
    """CNN block for feature extraction from spectrograms"""
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout2d(0.25)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = self.dropout(x)
        return x


class HybridCNNRNNAttention(nn.Module):
    """
    Hybrid CNN-RNN-Attention model for breath sound classification
    
    Architecture:
    1. CNN layers extract spatial features from spectrograms
    2. Bidirectional LSTM captures temporal patterns
    3. Attention mechanism focuses on important time steps
    4. Fully connected layers for classification
    """
    
    def __init__(
        self,
        input_channels: int = 1,
        cnn_channels: list = [64, 128, 256],
        rnn_hidden_size: int = 256,
        rnn_num_layers: int = 2,
        attention_dim: int = 128,
        num_classes: int = 4,
        dropout: float = 0.5
    ):
        super().__init__()
        
        self.input_channels = input_channels
        self.num_classes = num_classes
        
        # CNN Feature Extractor
        self.cnn_blocks = nn.ModuleList()
        in_ch = input_channels
        for out_ch in cnn_channels: 
            self.cnn_blocks.append(CNNBlock(in_ch, out_ch))
            in_ch = out_ch
        
        # Calculate CNN output size dynamically
        # We need to know the input size to calculate LSTM input_size
        # After 3 pooling layers (stride 2): height / 8, width / 8
        # For input (1, 168, time), after CNN: (cnn_channels[-1], 21, time/8)
        # LSTM input will be: cnn_channels[-1] * (168 // (2**len(cnn_channels)))
        self.cnn_output_channels = cnn_channels[-1]
        self.height_reduction = 2 ** len(cnn_channels)  # Each pooling divides by 2
        
        # For 168 frequency bins: 168 / 8 = 21
        # LSTM input size: 256 channels * 21 height = 5376
        # This is too large! Let's use adaptive pooling to reduce it
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, None))  # Reduce height to 4
        lstm_input_size = self.cnn_output_channels * 4
        
        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=lstm_input_size,
            hidden_size=rnn_hidden_size,
            num_layers=rnn_num_layers,
            batch_first=True,
            dropout=dropout if rnn_num_layers > 1 else 0,
            bidirectional=True
        )
        
        # Attention
        self.attention = AttentionLayer(
            hidden_dim=rnn_hidden_size * 2,  # *2 for bidirectional
            attention_dim=attention_dim
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(rnn_hidden_size * 2, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights for better convergence"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if 'weight_ih' in name:
                        nn.init.xavier_normal_(param)
                    elif 'weight_hh' in name:
                        nn.init.orthogonal_(param)
                    elif 'bias' in name:
                        nn.init.constant_(param, 0)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch, channels, height, width) - spectrogram
               Expected: (batch, 1, 168, time_frames)
        Returns:
            logits: (batch, num_classes)
            attention_weights: (batch, seq_len)
        """
        batch_size = x.size(0)
        
        # CNN feature extraction
        for cnn_block in self.cnn_blocks:
            x = cnn_block(x)
        
        # x shape after CNN: (batch, cnn_channels[-1], height, width)
        # Apply adaptive pooling to reduce height dimension
        x = self.adaptive_pool(x)  # (batch, channels, 4, width)
        
        # Reshape for RNN: use width as time dimension, flatten channels and height
        x = x.permute(0, 3, 1, 2)  # (batch, width, channels, height)
        x = x.reshape(batch_size, x.size(1), -1)  # (batch, time_steps, channels*height)
        
        # LSTM expects (batch, seq_len, input_size)
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, hidden_size * 2)
        
        # Attention
        context, attention_weights = self.attention(lstm_out)
        
        # Classification
        logits = self.classifier(context)
        
        return logits, attention_weights
    
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Get probability predictions"""
        logits, _ = self.forward(x)
        probas = F.softmax(logits, dim=1)
        return probas


def create_model(config: dict, device: torch.device) -> nn.Module:
    """Create model from configuration"""
    model_config = config['model']
    
    # Determine input channels based on features
    # mel_spectrogram (128) + mfcc (40) = 168
    input_channels = 1  # Will be reshaped in dataset
    
    model = HybridCNNRNNAttention(
        input_channels=input_channels,
        cnn_channels=model_config['cnn_channels'],
        rnn_hidden_size=model_config['rnn_hidden_size'],
        rnn_num_layers=model_config['rnn_num_layers'],
        attention_dim=model_config['attention_dim'],
        num_classes=model_config['num_classes'],
        dropout=model_config['dropout']
    )
    
    model = model.to(device)
    return model