"""
Utility functions for model training, evaluation, and visualization
"""

import os
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime

CLASS_NAMES = ['mel', 'nv', 'bkl', 'bcc', 'akiec', 'df', 'vasc']

RISK_MAP = {
    0: "High",    # mel
    3: "Medium",  # bcc
    4: "Medium",  # akiec
    1: "Low",
    2: "Low",
    5: "Low",
    6: "Low"
}


def get_risk(label, age_years):
    """
    Map class label and age (years) to a risk tier.
    """
    base = RISK_MAP[label]
    if age_years > 60 and base != "High":
        return "Medium"
    return base


def save_checkpoint(model, optimizer, epoch, accuracy, path):
    """
    Save model checkpoint
    
    Args:
        model: PyTorch model
        optimizer: PyTorch optimizer
        epoch: Current epoch
        accuracy: Validation accuracy
        path: Path to save checkpoint
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'accuracy': accuracy,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    torch.save(checkpoint, path)
    print(f"Checkpoint saved to {path}")


def load_checkpoint(path, model, optimizer=None):
    """
    Load model checkpoint
    
    Args:
        path: Path to checkpoint
        model: PyTorch model
        optimizer: PyTorch optimizer (optional)
    
    Returns:
        dict: Checkpoint dictionary
    """
    checkpoint = torch.load(path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    print(f"Checkpoint loaded from {path}")
    print(f"  Epoch: {checkpoint['epoch']}")
    print(f"  Accuracy: {checkpoint['accuracy']:.4f}")
    
    return checkpoint


def plot_training_history(history, save_path=None):
    """
    Plot training history (loss and accuracy curves)
    
    Args:
        history: Dictionary with training history
        save_path: Path to save plot (optional)
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Plot loss
    axes[0].plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    axes[0].plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(alpha=0.3)
    
    # Plot accuracy
    axes[1].plot(epochs, history['train_acc'], 'b-', label='Train Accuracy', linewidth=2)
    axes[1].plot(epochs, history['val_acc'], 'r-', label='Val Accuracy', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Accuracy', fontsize=12)
    axes[1].set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(alpha=0.3)
    
    # Plot F1 score
    axes[2].plot(epochs, history['val_f1'], 'g-', label='Val F1 Score', linewidth=2)
    axes[2].set_xlabel('Epoch', fontsize=12)
    axes[2].set_ylabel('F1 Score', fontsize=12)
    axes[2].set_title('Validation F1 Score', fontsize=14, fontweight='bold')
    axes[2].legend(fontsize=10)
    axes[2].grid(alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training history plot saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def count_parameters(model):
    """
    Count trainable parameters in model
    
    Args:
        model: PyTorch model
    
    Returns:
        int: Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_device():
    """
    Get available device (CUDA, MPS, or CPU)
    
    Returns:
        torch.device: Available device
    """
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')


def set_seed(seed=42):
    """
    Set random seed for reproducibility
    
    Args:
        seed (int): Random seed
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def print_model_summary(model, input_size=(3, 224, 224), metadata_size=2):
    """
    Print model summary
    
    Args:
        model: PyTorch model
        input_size: Input image size
        metadata_size: Metadata feature size
    """
    print("\n" + "="*60)
    print("MODEL SUMMARY")
    print("="*60)
    print(f"Model: {model.__class__.__name__}")
    print(f"Total Parameters: {count_parameters(model):,}")
    print(f"Input Image Size: {input_size}")
    print(f"Metadata Size: {metadata_size}")
    print("="*60 + "\n")


def calculate_class_weights(dataset, num_classes=7):
    """
    Calculate class weights for imbalanced dataset
    
    Args:
        dataset: PyTorch dataset
        num_classes: Number of classes
    
    Returns:
        torch.Tensor: Class weights
    """
    labels = []
    for i in range(len(dataset)):
        labels.append(dataset[i][3].item())
    
    labels = np.array(labels)
    class_counts = np.bincount(labels, minlength=num_classes)
    
    # Calculate weights (inverse frequency)
    weights = 1.0 / (class_counts + 1e-6)
    weights = weights / weights.sum() * num_classes  # Normalize
    
    return torch.FloatTensor(weights)


def visualize_predictions(images, true_labels, pred_labels, class_names, 
                          num_images=16, save_path=None):
    """
    Visualize model predictions
    
    Args:
        images: Batch of images
        true_labels: True labels
        pred_labels: Predicted labels
        class_names: List of class names
        num_images: Number of images to display
        save_path: Path to save visualization
    """
    num_images = min(num_images, len(images))
    rows = int(np.sqrt(num_images))
    cols = int(np.ceil(num_images / rows))
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols*3, rows*3))
    axes = axes.flatten()
    
    for idx in range(num_images):
        img = images[idx].cpu().numpy().transpose(1, 2, 0)
        # Denormalize
        img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        img = np.clip(img, 0, 1)
        
        axes[idx].imshow(img)
        axes[idx].axis('off')
        
        true_class = class_names[true_labels[idx]]
        pred_class = class_names[pred_labels[idx]]
        
        color = 'green' if true_labels[idx] == pred_labels[idx] else 'red'
        axes[idx].set_title(f'True: {true_class}\nPred: {pred_class}', 
                           color=color, fontsize=9)
    
    # Hide extra subplots
    for idx in range(num_images, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Predictions visualization saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def log_experiment(config, results, log_file='experiments.log'):
    """
    Log experiment configuration and results
    
    Args:
        config: Configuration dictionary
        results: Results dictionary
        log_file: Path to log file
    """
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    with open(log_file, 'a') as f:
        f.write(f"\n{'='*60}\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"{'='*60}\n")
        f.write("Configuration:\n")
        for key, value in config.items():
            f.write(f"  {key}: {value}\n")
        f.write("\nResults:\n")
        for key, value in results.items():
            if isinstance(value, (int, float, str)):
                f.write(f"  {key}: {value}\n")
        f.write(f"{'='*60}\n")
    
    print(f"Experiment logged to {log_file}")
