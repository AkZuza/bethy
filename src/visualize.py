"""
Visualization utilities for breath sound classification
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
from typing import Dict, List, Optional
import os


def plot_spectrogram(
    spectrogram: np.ndarray,
    title: str = "Spectrogram",
    save_path: Optional[str] = None,
    figsize: tuple = (10, 4)
):
    """
    Plot spectrogram
    
    Args:
        spectrogram: 2D array of spectrogram values
        title: Plot title
        save_path: Path to save figure (optional)
        figsize: Figure size
    """
    plt.figure(figsize=figsize)
    plt.imshow(spectrogram, aspect='auto', origin='lower', cmap='viridis')
    plt.colorbar(format='%+2.0f dB')
    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel('Frequency')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_attention_weights(
    attention_weights: np.ndarray,
    title: str = "Attention Weights",
    save_path: Optional[str] = None,
    figsize: tuple = (12, 3)
):
    """
    Plot attention weights over time
    
    Args:
        attention_weights: 1D array of attention weights
        title: Plot title
        save_path: Path to save figure (optional)
        figsize: Figure size
    """
    plt.figure(figsize=figsize)
    plt.bar(range(len(attention_weights)), attention_weights, color='steelblue')
    plt.title(title)
    plt.xlabel('Time Step')
    plt.ylabel('Attention Weight')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_attention_on_spectrogram(
    spectrogram: np.ndarray,
    attention_weights: np.ndarray,
    title: str = "Attention on Spectrogram",
    save_path: Optional[str] = None,
    figsize: tuple = (12, 5)
):
    """
    Overlay attention weights on spectrogram
    
    Args:
        spectrogram: 2D array of spectrogram values
        attention_weights: 1D array of attention weights
        title: Plot title
        save_path: Path to save figure (optional)
        figsize: Figure size
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, 
                                     gridspec_kw={'height_ratios': [3, 1]})
    
    # Plot spectrogram
    im = ax1.imshow(spectrogram, aspect='auto', origin='lower', cmap='viridis')
    ax1.set_title(title)
    ax1.set_ylabel('Frequency')
    plt.colorbar(im, ax=ax1, format='%+2.0f dB')
    
    # Plot attention weights
    time_steps = np.linspace(0, spectrogram.shape[1], len(attention_weights))
    ax2.bar(time_steps, attention_weights, width=spectrogram.shape[1]/len(attention_weights),
            color='steelblue', alpha=0.7)
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Attention')
    ax2.set_xlim(0, spectrogram.shape[1])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    title: str = "Confusion Matrix",
    save_path: Optional[str] = None,
    figsize: tuple = (8, 6),
    normalize: bool = False
):
    """
    Plot confusion matrix
    
    Args:
        cm: Confusion matrix
        class_names: List of class names
        title: Plot title
        save_path: Path to save figure (optional)
        figsize: Figure size
        normalize: Whether to normalize values
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
        fmt = '.2f'
    else:
        fmt = 'd'
    
    plt.figure(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count' if not normalize else 'Proportion'})
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_training_curves(
    history: Dict[str, List[float]],
    metrics: List[str] = ['loss', 'accuracy'],
    save_path: Optional[str] = None,
    figsize: tuple = (12, 4)
):
    """
    Plot training and validation curves
    
    Args:
        history: Dictionary with 'train' and 'val' keys containing metrics
        metrics: List of metrics to plot
        save_path: Path to save figure (optional)
        figsize: Figure size
    """
    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=figsize)
    
    if n_metrics == 1:
        axes = [axes]
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        
        train_key = f'train_{metric}'
        val_key = f'val_{metric}'
        
        if train_key in history:
            epochs = range(1, len(history[train_key]) + 1)
            ax.plot(epochs, history[train_key], 'b-', label=f'Train {metric}')
        
        if val_key in history:
            epochs = range(1, len(history[val_key]) + 1)
            ax.plot(epochs, history[val_key], 'r-', label=f'Val {metric}')
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel(metric.capitalize())
        ax.set_title(f'{metric.capitalize()} over Epochs')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_class_distribution(
    labels: np.ndarray,
    class_names: List[str],
    title: str = "Class Distribution",
    save_path: Optional[str] = None,
    figsize: tuple = (8, 5)
):
    """
    Plot class distribution
    
    Args:
        labels: Array of class labels
        class_names: List of class names
        title: Plot title
        save_path: Path to save figure (optional)
        figsize: Figure size
    """
    unique, counts = np.unique(labels, return_counts=True)
    
    plt.figure(figsize=figsize)
    bars = plt.bar([class_names[i] for i in unique], counts, color='steelblue')
    plt.title(title)
    plt.xlabel('Class')
    plt.ylabel('Count')
    
    # Add count labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_prediction_confidence(
    probabilities: np.ndarray,
    class_names: List[str],
    true_label: Optional[int] = None,
    predicted_label: Optional[int] = None,
    title: str = "Prediction Confidence",
    save_path: Optional[str] = None,
    figsize: tuple = (8, 5)
):
    """
    Plot prediction confidence as bar chart
    
    Args:
        probabilities: Array of class probabilities
        class_names: List of class names
        true_label: True class label (optional)
        predicted_label: Predicted class label (optional)
        title: Plot title
        save_path: Path to save figure (optional)
        figsize: Figure size
    """
    colors = ['steelblue'] * len(probabilities)
    
    if predicted_label is not None:
        colors[predicted_label] = 'green' if true_label == predicted_label else 'red'
    
    if true_label is not None and true_label != predicted_label:
        colors[true_label] = 'orange'
    
    plt.figure(figsize=figsize)
    bars = plt.bar(class_names, probabilities, color=colors)
    plt.title(title)
    plt.xlabel('Class')
    plt.ylabel('Probability')
    plt.ylim(0, 1)
    
    # Add percentage labels on bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{probabilities[i]*100:.1f}%',
                ha='center', va='bottom')
    
    # Add legend
    legend_elements = []
    if predicted_label is not None:
        legend_elements.append(plt.Rectangle((0,0),1,1, fc='green' if true_label == predicted_label else 'red',
                                            label='Predicted'))
    if true_label is not None and true_label != predicted_label:
        legend_elements.append(plt.Rectangle((0,0),1,1, fc='orange', label='True'))
    
    if legend_elements:
        plt.legend(handles=legend_elements)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def save_visualization_grid(
    spectrograms: List[np.ndarray],
    attention_weights: List[np.ndarray],
    labels: List[int],
    predictions: List[int],
    class_names: List[str],
    save_path: str,
    max_samples: int = 16
):
    """
    Create a grid of spectrogram + attention visualizations
    
    Args:
        spectrograms: List of spectrograms
        attention_weights: List of attention weight arrays
        labels: List of true labels
        predictions: List of predicted labels
        class_names: List of class names
        save_path: Path to save the grid
        max_samples: Maximum number of samples to show
    """
    n_samples = min(len(spectrograms), max_samples)
    n_cols = 4
    n_rows = (n_samples + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*4, n_rows*3))
    axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes
    
    for idx in range(n_samples):
        ax = axes[idx]
        
        # Plot spectrogram
        im = ax.imshow(spectrograms[idx], aspect='auto', origin='lower', cmap='viridis')
        
        # Overlay attention weights
        if attention_weights[idx] is not None:
            time_steps = np.linspace(0, spectrograms[idx].shape[1], len(attention_weights[idx]))
            ax_twin = ax.twinx()
            ax_twin.plot(time_steps, attention_weights[idx], 'r-', alpha=0.7, linewidth=2)
            ax_twin.set_ylim(0, max(attention_weights[idx]) * 1.2)
            ax_twin.axis('off')
        
        # Title with prediction info
        true_class = class_names[labels[idx]]
        pred_class = class_names[predictions[idx]]
        color = 'green' if labels[idx] == predictions[idx] else 'red'
        ax.set_title(f'True: {true_class}\nPred: {pred_class}', color=color, fontsize=8)
        ax.axis('off')
    
    # Hide unused subplots
    for idx in range(n_samples, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
