"""
Main entry point for Skin Cancer Detection Project
Provides a unified interface for training, evaluation, and optimization
"""

import argparse
import os
import sys

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.train import train
from src.evaluate import evaluate
from src.aco import run_aco_optimization
from src.utils import set_seed


def main():
    """
    Main function with command-line interface
    """
    parser = argparse.ArgumentParser(
        description='Skin Cancer Detection using Multimodal Deep Learning',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train a model
  python main.py --mode train --backbone resnet50 --epochs 30
  
  # Evaluate a trained model
  python main.py --mode evaluate --model_path models/best_model.pth
  
  # Run hyperparameter optimization with ACO
  python main.py --mode optimize
  
  # Train with custom parameters
  python main.py --mode train --batch_size 64 --lr 0.0001 --backbone efficientnet_b0
        """
    )
    
    # Mode selection
    parser.add_argument(
        '--mode', type=str, required=True,
        choices=['train', 'evaluate', 'optimize'],
        help='Operation mode: train, evaluate, or optimize'
    )
    
    # Data arguments
    parser.add_argument(
        '--data_dir', type=str, default='data/images',
        help='Directory containing images (default: data/images)'
    )
    parser.add_argument(
        '--metadata_file', type=str, default='data/metadata.csv',
        help='Path to metadata CSV file (default: data/metadata.csv)'
    )
    
    # Model arguments
    parser.add_argument(
        '--model_type', type=str, default='multimodal',
        choices=['multimodal', 'image_only'],
        help='Model type (default: multimodal)'
    )
    parser.add_argument(
        '--backbone', type=str, default='resnet50',
        choices=['resnet50', 'efficientnet_b0', 'mobilenet_v2'],
        help='CNN backbone architecture (default: resnet50)'
    )
    parser.add_argument(
        '--pretrained', action='store_true', default=True,
        help='Use pretrained ImageNet weights (default: True)'
    )
    parser.add_argument(
        '--num_classes', type=int, default=7,
        help='Number of output classes (default: 7)'
    )
    
    # Training arguments
    parser.add_argument(
        '--batch_size', type=int, default=32,
        help='Batch size for training (default: 32)'
    )
    parser.add_argument(
        '--epochs', type=int, default=30,
        help='Number of training epochs (default: 30)'
    )
    parser.add_argument(
        '--lr', type=float, default=0.001,
        help='Learning rate (default: 0.001)'
    )
    parser.add_argument(
        '--num_workers', type=int, default=4,
        help='Number of data loading workers (default: 4)'
    )
    
    # Save/Load arguments
    parser.add_argument(
        '--save_dir', type=str, default='models',
        help='Directory to save models (default: models)'
    )
    parser.add_argument(
        '--model_path', type=str, default='models/best_model.pth',
        help='Path to model checkpoint for evaluation (default: models/best_model.pth)'
    )
    parser.add_argument(
        '--results_dir', type=str, default='results',
        help='Directory to save results (default: results)'
    )
    
    # Other arguments
    parser.add_argument(
        '--seed', type=int, default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    
    args = parser.parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Create necessary directories
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)
    
    # Print configuration
    print("\n" + "="*70)
    print("SKIN CANCER DETECTION - Multimodal Deep Learning")
    print("="*70)
    print(f"Mode: {args.mode.upper()}")
    print(f"Model Type: {args.model_type}")
    print(f"Backbone: {args.backbone}")
    print(f"Data Directory: {args.data_dir}")
    print("="*70 + "\n")
    
    # Execute based on mode
    if args.mode == 'train':
        print("Starting training...\n")
        
        config = {
            'data_dir': args.data_dir,
            'metadata_file': args.metadata_file,
            'save_dir': args.save_dir,
            'model_type': args.model_type,
            'backbone': args.backbone,
            'num_classes': args.num_classes,
            'pretrained': args.pretrained,
            'batch_size': args.batch_size,
            'num_epochs': args.epochs,
            'learning_rate': args.lr,
            'num_workers': args.num_workers
        }
        
        model, history = train(config)
        
        print("\n" + "="*70)
        print("Training completed successfully!")
        print(f"Best model saved to: {os.path.join(args.save_dir, 'best_model.pth')}")
        print("="*70 + "\n")
    
    elif args.mode == 'evaluate':
        print("Starting evaluation...\n")
        
        # Check if model exists
        if not os.path.exists(args.model_path):
            print(f"Error: Model file not found at {args.model_path}")
            print("Please provide a valid model path using --model_path")
            sys.exit(1)
        
        config = {
            'data_dir': args.data_dir,
            'metadata_file': args.metadata_file,
            'model_path': args.model_path,
            'results_dir': args.results_dir,
            'model_type': args.model_type,
            'backbone': args.backbone,
            'num_classes': args.num_classes,
            'batch_size': args.batch_size,
            'num_workers': args.num_workers
        }
        
        results = evaluate(config)
        
        print("\n" + "="*70)
        print("Evaluation completed successfully!")
        print(f"Results saved to: {args.results_dir}")
        print("="*70 + "\n")
    
    elif args.mode == 'optimize':
        print("Starting hyperparameter optimization with ACO...\n")
        print("Note: This may take several hours depending on search space.")
        print("Consider reducing n_ants and n_iterations for faster results.\n")
        
        best_params, best_score, history = run_aco_optimization()
        
        print("\n" + "="*70)
        print("Optimization completed successfully!")
        print(f"Best parameters: {best_params}")
        print(f"Best score: {best_score:.4f}")
        print("="*70 + "\n")


def quick_train():
    """
    Quick training function with default parameters (for direct import usage)
    """
    config = {
        'data_dir': 'data/images',
        'metadata_file': 'data/metadata.csv',
        'save_dir': 'models',
        'model_type': 'multimodal',
        'backbone': 'resnet50',
        'num_classes': 7,
        'pretrained': True,
        'batch_size': 32,
        'num_epochs': 30,
        'learning_rate': 0.001,
        'num_workers': 4
    }
    
    set_seed(42)
    os.makedirs(config['save_dir'], exist_ok=True)
    
    print("Quick training with default configuration...")
    model, history = train(config)
    return model, history


def quick_evaluate(model_path='models/best_model.pth'):
    """
    Quick evaluation function (for direct import usage)
    """
    config = {
        'data_dir': 'data/images',
        'metadata_file': 'data/metadata.csv',
        'model_path': model_path,
        'results_dir': 'results',
        'model_type': 'multimodal',
        'backbone': 'resnet50',
        'num_classes': 7,
        'batch_size': 32,
        'num_workers': 4
    }
    
    set_seed(42)
    os.makedirs(config['results_dir'], exist_ok=True)
    
    print("Quick evaluation...")
    results = evaluate(config)
    return results


if __name__ == "__main__":
    main()
