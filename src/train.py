"""
Training script for Skin Cancer Detection Model
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report

from dataset import SkinCancerDataset, get_transforms
from model import get_model
from utils import save_checkpoint, load_checkpoint, plot_training_history


def train_one_epoch(model, dataloader, criterion, optimizer, device, model_type, num_classes):
    """
    Train for one epoch
    
    Returns:
        float: Average training loss
        float: Training accuracy
    """
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    for images, ages, genders, labels in tqdm(dataloader, desc="Training"):
        images = images.to(device)
        ages = ages.to(device)
        genders = genders.to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        if model_type == 'multimodal':
            metadata = torch.stack([ages, genders], dim=1)
            outputs = model(images, metadata)
        else:
            outputs = model(images)

        if num_classes == 1:
            labels = labels.float().to(device).view(-1, 1)
        else:
            labels = labels.to(device)

        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item() * images.size(0)
        if num_classes == 1:
            probs = torch.sigmoid(outputs)
            predicted = (probs >= 0.5).long().view(-1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.view(-1).cpu().numpy())
        else:
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)
    
    return epoch_loss, epoch_acc


def validate(model, dataloader, criterion, device, model_type, num_classes):
    """
    Validate the model
    
    Returns:
        float: Validation loss
        float: Validation accuracy
        float: Validation F1 score
    """
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, ages, genders, labels in tqdm(dataloader, desc="Validation"):
            images = images.to(device)
            ages = ages.to(device)
            genders = genders.to(device)
            
            # Forward pass
            if model_type == 'multimodal':
                metadata = torch.stack([ages, genders], dim=1)
                outputs = model(images, metadata)
            else:
                outputs = model(images)

            if num_classes == 1:
                labels = labels.float().to(device).view(-1, 1)
            else:
                labels = labels.to(device)

            loss = criterion(outputs, labels)
            
            # Statistics
            running_loss += loss.item() * images.size(0)
            if num_classes == 1:
                probs = torch.sigmoid(outputs)
                predicted = (probs >= 0.5).long().view(-1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.view(-1).cpu().numpy())
            else:
                _, predicted = torch.max(outputs, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
    
    val_loss = running_loss / len(dataloader.dataset)
    val_acc = accuracy_score(all_labels, all_preds)
    val_f1 = f1_score(all_labels, all_preds, average='weighted')
    
    return val_loss, val_acc, val_f1


def train(config):
    """
    Main training function
    
    Args:
        config (dict): Configuration dictionary with training parameters
    """
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load dataset
    print("Loading dataset...")
    train_transform = get_transforms(mode='train')
    val_transform = get_transforms(mode='val')
    
    full_dataset = SkinCancerDataset(
        data_dir=config['data_dir'],
        metadata_file=config['metadata_file'],
        transform=train_transform
    )
    
    # Split dataset
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    val_dataset.dataset.transform = val_transform
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers']
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers']
    )
    
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    
    # Create model
    print("Creating model...")
    model = get_model(
        model_type=config['model_type'],
        num_classes=config['num_classes'],
        backbone=config['backbone'],
        pretrained=config['pretrained']
    ).to(device)
    
    # Loss and optimizer
    criterion = nn.BCEWithLogitsLoss() if config['num_classes'] == 1 else nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=3, factor=0.5, verbose=True
    )
    
    # Training loop
    print("Starting training...")
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [], 'val_f1': []
    }
    
    best_val_acc = 0.0
    
    for epoch in range(config['num_epochs']):
        print(f"\nEpoch {epoch+1}/{config['num_epochs']}")
        print("-" * 50)
        
        # Train
        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            config['model_type'],
            config['num_classes']
        )
        
        # Validate
        val_loss, val_acc, val_f1 = validate(
            model,
            val_loader,
            criterion,
            device,
            config['model_type'],
            config['num_classes']
        )
        
        # Update scheduler
        scheduler.step(val_loss)
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_f1'].append(val_f1)
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(
                model, optimizer, epoch, val_acc,
                os.path.join(config['save_dir'], 'best_model.pth')
            )
            print(f"✓ Saved best model with val_acc: {val_acc:.4f}")
    
    # Save final model
    save_checkpoint(
        model, optimizer, config['num_epochs'], val_acc,
        os.path.join(config['save_dir'], 'final_model.pth')
    )
    
    # Plot training history
    plot_training_history(history, save_path=os.path.join(config['save_dir'], 'training_history.png'))
    
    print("\nTraining completed!")
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    
    return model, history


if __name__ == "__main__":
    # Training configuration
    config = {
        'data_dir': '/content/images',
        'metadata_file': '../data/metadata.csv',
        'save_dir': '../models',
        'model_type': 'multimodal',  # 'multimodal' or 'image_only'
        'backbone': 'resnet50',       # 'resnet50', 'efficientnet_b0', 'mobilenet_v2'
        'num_classes': 7,
        'pretrained': True,
        'batch_size': 32,
        'num_epochs': 30,
        'learning_rate': 0.001,
        'num_workers': 4
    }
    
    # Create save directory
    os.makedirs(config['save_dir'], exist_ok=True)
    
    # Train model
    model, history = train(config)
