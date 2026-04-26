"""
Evaluation script for Skin Cancer Detection Model
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from sklearn.metrics import (
    classification_report, confusion_matrix, 
    accuracy_score, precision_recall_fscore_support,
    roc_auc_score, roc_curve
)
from tqdm import tqdm

from dataset import SkinCancerDataset, get_transforms
from model import get_model
from utils import load_checkpoint


def evaluate_model(model, dataloader, device, class_names):
    """
    Evaluate model and compute metrics
    
    Returns:
        dict: Dictionary containing evaluation metrics
    """
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            images = batch['image'].to(device)
            metadata = batch['metadata'].to(device)
            labels = batch['label'].to(device)
            
            # Forward pass
            outputs = model(images, metadata)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # Compute metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, support = precision_recall_fscore_support(
        all_labels, all_preds, average='weighted'
    )
    
    # Classification report
    report = classification_report(
        all_labels, all_preds, 
        target_names=class_names,
        output_dict=True
    )
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    results = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'predictions': all_preds,
        'labels': all_labels,
        'probabilities': all_probs,
        'confusion_matrix': cm,
        'classification_report': report
    }
    
    return results


def plot_confusion_matrix(cm, class_names, save_path):
    """
    Plot confusion matrix
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"Confusion matrix saved to {save_path}")
    plt.close()


def plot_roc_curves(labels, probs, class_names, save_path):
    """
    Plot ROC curves for each class
    """
    n_classes = len(class_names)
    
    plt.figure(figsize=(12, 10))
    
    for i in range(n_classes):
        # Binarize labels for one-vs-rest
        binary_labels = (labels == i).astype(int)
        class_probs = probs[:, i]
        
        # Compute ROC curve
        fpr, tpr, _ = roc_curve(binary_labels, class_probs)
        auc = roc_auc_score(binary_labels, class_probs)
        
        plt.plot(fpr, tpr, label=f'{class_names[i]} (AUC = {auc:.3f})')
    
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves - One vs Rest')
    plt.legend(loc='lower right')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"ROC curves saved to {save_path}")
    plt.close()


def print_report(results, class_names):
    """
    Print evaluation report
    """
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"Overall Accuracy: {results['accuracy']:.4f}")
    print(f"Weighted Precision: {results['precision']:.4f}")
    print(f"Weighted Recall: {results['recall']:.4f}")
    print(f"Weighted F1-Score: {results['f1_score']:.4f}")
    print("\n" + "-"*60)
    print("Per-Class Metrics:")
    print("-"*60)
    
    for class_name in class_names:
        metrics = results['classification_report'][class_name]
        print(f"\n{class_name.upper()}:")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall: {metrics['recall']:.4f}")
        print(f"  F1-Score: {metrics['f1-score']:.4f}")
        print(f"  Support: {int(metrics['support'])}")
    
    print("\n" + "="*60)


def evaluate(config):
    """
    Main evaluation function
    """
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Class names
    class_names = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
    
    # Load dataset
    print("Loading test dataset...")
    test_transform = get_transforms(mode='val')
    test_dataset = SkinCancerDataset(
        data_dir=config['data_dir'],
        metadata_file=config['metadata_file'],
        transform=test_transform
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers']
    )
    
    print(f"Test samples: {len(test_dataset)}")
    
    # Create model
    print("Loading model...")
    model = get_model(
        model_type=config['model_type'],
        num_classes=config['num_classes'],
        backbone=config['backbone'],
        pretrained=False
    ).to(device)
    
    # Load checkpoint
    checkpoint = load_checkpoint(config['model_path'], model)
    print(f"Loaded model from {config['model_path']}")
    
    # Evaluate
    results = evaluate_model(model, test_loader, device, class_names)
    
    # Print report
    print_report(results, class_names)
    
    # Create results directory
    os.makedirs(config['results_dir'], exist_ok=True)
    
    # Plot and save confusion matrix
    plot_confusion_matrix(
        results['confusion_matrix'],
        class_names,
        os.path.join(config['results_dir'], 'confusion_matrix.png')
    )
    
    # Plot and save ROC curves
    plot_roc_curves(
        results['labels'],
        results['probabilities'],
        class_names,
        os.path.join(config['results_dir'], 'roc_curves.png')
    )
    
    # Save detailed report to file
    report_path = os.path.join(config['results_dir'], 'evaluation_report.txt')
    with open(report_path, 'w') as f:
        f.write("="*60 + "\n")
        f.write("EVALUATION RESULTS\n")
        f.write("="*60 + "\n")
        f.write(f"Overall Accuracy: {results['accuracy']:.4f}\n")
        f.write(f"Weighted Precision: {results['precision']:.4f}\n")
        f.write(f"Weighted Recall: {results['recall']:.4f}\n")
        f.write(f"Weighted F1-Score: {results['f1_score']:.4f}\n\n")
        
        for class_name in class_names:
            metrics = results['classification_report'][class_name]
            f.write(f"\n{class_name.upper()}:\n")
            f.write(f"  Precision: {metrics['precision']:.4f}\n")
            f.write(f"  Recall: {metrics['recall']:.4f}\n")
            f.write(f"  F1-Score: {metrics['f1-score']:.4f}\n")
            f.write(f"  Support: {int(metrics['support'])}\n")
    
    print(f"\nDetailed report saved to {report_path}")
    
    return results


if __name__ == "__main__":
    # Evaluation configuration
    config = {
        'data_dir': '/content/images',
        'metadata_file': '/content/cleaned_metadata.csv',
        'model_path': '../models/best_model.pth',
        'results_dir': '../results',
        'model_type': 'multimodal',
        'backbone': 'resnet50',
        'num_classes': 7,
        'batch_size': 32,
        'num_workers': 4
    }
    
    # Evaluate model
    results = evaluate(config)
