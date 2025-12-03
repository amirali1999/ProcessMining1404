"""
Utility functions for the prediction engine
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any
import matplotlib.pyplot as plt
import seaborn as sns


def plot_training_history(history, save_path=None):
    """
    Plot training history from Keras
    
    Args:
        history: Keras History object
        save_path: Path to save plot (optional)
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss
    axes[0].plot(history.history['loss'], label='Train Loss')
    if 'val_loss' in history.history:
        axes[0].plot(history.history['val_loss'], label='Val Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    # Accuracy/MAE
    metric_key = 'accuracy' if 'accuracy' in history.history else 'mae'
    axes[1].plot(history.history[metric_key], label=f'Train {metric_key}')
    if f'val_{metric_key}' in history.history:
        axes[1].plot(history.history[f'val_{metric_key}'], label=f'Val {metric_key}')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel(metric_key.upper())
    axes[1].set_title(f'Training and Validation {metric_key.upper()}')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
    else:
        plt.show()


def plot_confusion_matrix(cm, class_names, save_path=None):
    """
    Plot confusion matrix
    
    Args:
        cm: Confusion matrix
        class_names: List of class names
        save_path: Path to save plot (optional)
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
    else:
        plt.show()


def plot_feature_importance(feature_importance_df, top_n=20, save_path=None):
    """
    Plot feature importance
    
    Args:
        feature_importance_df: DataFrame with 'feature' and 'importance' columns
        top_n: Number of top features to show
        save_path: Path to save plot (optional)
    """
    top_features = feature_importance_df.head(top_n)
    
    plt.figure(figsize=(10, 8))
    plt.barh(range(len(top_features)), top_features['importance'])
    plt.yticks(range(len(top_features)), top_features['feature'])
    plt.xlabel('Importance')
    plt.title(f'Top {top_n} Feature Importances')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
    else:
        plt.show()


def analyze_case(case_data: Dict[str, Any]):
    """
    Analyze a single case and print statistics
    
    Args:
        case_data: Dictionary with case information
    """
    print("="*60)
    print("CASE ANALYSIS")
    print("="*60)
    print(f"Case ID: {case_data['case_id']}")
    print(f"Number of activities: {len(case_data['activities'])}")
    print(f"Activities: {' → '.join(case_data['activities'])}")
    
    if 'timestamps' in case_data:
        timestamps = case_data['timestamps']
        duration = (timestamps[-1] - timestamps[0]).total_seconds()
        print(f"Duration: {duration:.2f} seconds ({duration/3600:.2f} hours)")
        
        # Calculate inter-event times
        inter_event_times = []
        for i in range(1, len(timestamps)):
            inter_event_times.append((timestamps[i] - timestamps[i-1]).total_seconds())
        
        if inter_event_times:
            print(f"Average time between events: {np.mean(inter_event_times):.2f} seconds")
            print(f"Min time between events: {np.min(inter_event_times):.2f} seconds")
            print(f"Max time between events: {np.max(inter_event_times):.2f} seconds")
    
    print("="*60)


def calculate_process_metrics(df: pd.DataFrame):
    """
    Calculate overall process metrics from event log
    
    Args:
        df: Event log DataFrame
        
    Returns:
        Dictionary with metrics
    """
    metrics = {}
    
    # Basic counts
    metrics['num_cases'] = df['case:concept:name'].nunique()
    metrics['num_events'] = len(df)
    metrics['num_activities'] = df['concept:name'].nunique()
    
    # Case statistics
    case_lengths = df.groupby('case:concept:name').size()
    metrics['avg_case_length'] = case_lengths.mean()
    metrics['min_case_length'] = case_lengths.min()
    metrics['max_case_length'] = case_lengths.max()
    metrics['std_case_length'] = case_lengths.std()
    
    # Activity statistics
    activity_counts = df['concept:name'].value_counts()
    metrics['most_common_activity'] = activity_counts.index[0]
    metrics['most_common_activity_count'] = activity_counts.iloc[0]
    
    # Time statistics (if timestamps available)
    if 'time:timestamp' in df.columns:
        case_durations = df.groupby('case:concept:name')['time:timestamp'].agg(
            lambda x: (x.max() - x.min()).total_seconds()
        )
        metrics['avg_case_duration'] = case_durations.mean()
        metrics['min_case_duration'] = case_durations.min()
        metrics['max_case_duration'] = case_durations.max()
    
    return metrics


def print_metrics(metrics: Dict[str, Any]):
    """Print metrics in a formatted way"""
    print("\n" + "="*60)
    print("PROCESS METRICS")
    print("="*60)
    
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key}: {value:.2f}")
        else:
            print(f"{key}: {value}")
    
    print("="*60 + "\n")


def export_predictions_to_csv(predictions: List[Dict], filepath: str):
    """
    Export predictions to CSV file
    
    Args:
        predictions: List of prediction dictionaries
        filepath: Output CSV file path
    """
    df = pd.DataFrame(predictions)
    df.to_csv(filepath, index=False)
    print(f"Predictions exported to {filepath}")


def compare_models(results: Dict[str, Dict[str, float]]):
    """
    Compare multiple models and print comparison table
    
    Args:
        results: Dictionary with model names as keys and metrics as values
    """
    print("\n" + "="*60)
    print("MODEL COMPARISON")
    print("="*60)
    
    comparison_df = pd.DataFrame(results).T
    print(comparison_df.to_string())
    
    # Find best model for each metric
    print("\nBest Models:")
    for metric in comparison_df.columns:
        best_model = comparison_df[metric].idxmax()
        best_value = comparison_df[metric].max()
        print(f"  {metric}: {best_model} ({best_value:.4f})")
    
    print("="*60 + "\n")
