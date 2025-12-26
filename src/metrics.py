"""
Evaluation metrics for breath sound classification
"""

import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from typing import Dict, List, Tuple


class MetricsCalculator:
    """Calculate evaluation metrics for ICBHI dataset"""
    
    def __init__(self, num_classes: int = 4):
        self.num_classes = num_classes
        self.class_names = ['normal', 'wheeze', 'crackle', 'both']
    
    def calculate_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: np.ndarray = None
    ) -> Dict[str, float]:
        """
        Calculate comprehensive metrics
        
        Args:
            y_true: Ground truth labels
            y_pred: Predicted labels
            y_proba: Prediction probabilities (optional)
        
        Returns:
            Dictionary of metrics
        """
        metrics = {}
        
        # Overall accuracy
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        
        # Per-class metrics
        precision = precision_score(y_true, y_pred, average=None, zero_division=0)
        recall = recall_score(y_true, y_pred, average=None, zero_division=0)
        f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
        
        for i, class_name in enumerate(self.class_names):
            metrics[f'precision_{class_name}'] = precision[i]
            metrics[f'recall_{class_name}'] = recall[i]
            metrics[f'f1_{class_name}'] = f1[i]
        
        # Macro-averaged metrics
        metrics['precision_macro'] = precision_score(y_true, y_pred, average='macro', zero_division=0)
        metrics['recall_macro'] = recall_score(y_true, y_pred, average='macro', zero_division=0)
        metrics['f1_macro'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
        
        # Weighted-averaged metrics
        metrics['precision_weighted'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics['recall_weighted'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics['f1_weighted'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        # Specificity (per-class)
        cm = confusion_matrix(y_true, y_pred, labels=range(self.num_classes))
        specificity = self._calculate_specificity(cm)
        
        for i, class_name in enumerate(self.class_names):
            metrics[f'specificity_{class_name}'] = specificity[i]
        
        metrics['specificity_macro'] = np.mean(specificity)
        
        # ICBHI Score (average of sensitivity and specificity)
        metrics['icbhi_score'] = (metrics['recall_macro'] + metrics['specificity_macro']) / 2
        
        # Confusion matrix
        metrics['confusion_matrix'] = cm
        
        return metrics
    
    def _calculate_specificity(self, cm: np.ndarray) -> np.ndarray:
        """
        Calculate specificity for each class
        
        Specificity = TN / (TN + FP)
        """
        specificity = np.zeros(self.num_classes)
        
        for i in range(self.num_classes):
            # True Negatives: all correct predictions excluding class i
            tn = np.sum(cm) - np.sum(cm[i, :]) - np.sum(cm[:, i]) + cm[i, i]
            # False Positives: predictions as class i that were not class i
            fp = np.sum(cm[:, i]) - cm[i, i]
            
            if (tn + fp) > 0:
                specificity[i] = tn / (tn + fp)
            else:
                specificity[i] = 0.0
        
        return specificity
    
    def get_classification_report(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> str:
        """Get detailed classification report"""
        return classification_report(
            y_true,
            y_pred,
            target_names=self.class_names,
            zero_division=0
        )
    
    def print_metrics(self, metrics: Dict[str, float], prefix: str = ""):
        """Print metrics in a formatted way"""
        print(f"\n{prefix}Metrics:")
        print(f"{'='*60}")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"ICBHI Score: {metrics['icbhi_score']:.4f}")
        print(f"\nMacro-averaged:")
        print(f"  Precision: {metrics['precision_macro']:.4f}")
        print(f"  Recall (Sensitivity): {metrics['recall_macro']:.4f}")
        print(f"  F1 Score: {metrics['f1_macro']:.4f}")
        print(f"  Specificity: {metrics['specificity_macro']:.4f}")
        print(f"\nPer-class metrics:")
        for class_name in self.class_names:
            print(f"  {class_name.capitalize()}:")
            print(f"    Precision: {metrics[f'precision_{class_name}']:.4f}")
            print(f"    Recall: {metrics[f'recall_{class_name}']:.4f}")
            print(f"    F1: {metrics[f'f1_{class_name}']:.4f}")
            print(f"    Specificity: {metrics[f'specificity_{class_name}']:.4f}")
        print(f"{'='*60}\n")


def compute_loss_weights(dataset) -> torch.Tensor:
    """Compute class weights for weighted loss"""
    return dataset.class_weights


class EarlyStopping:
    """Early stopping to prevent overfitting"""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.0, mode: str = 'min'):
        """
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
            mode: 'min' for loss, 'max' for metrics like accuracy
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
    def __call__(self, score: float) -> bool:
        """
        Check if training should stop
        
        Args:
            score: Current metric value
        
        Returns:
            True if should stop, False otherwise
        """
        if self.best_score is None:
            self.best_score = score
            return False
        
        if self.mode == 'min':
            improved = score < (self.best_score - self.min_delta)
        else:  # mode == 'max'
            improved = score > (self.best_score + self.min_delta)
        
        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                return True
        
        return False
    
    def reset(self):
        """Reset early stopping"""
        self.counter = 0
        self.best_score = None
        self.early_stop = False
