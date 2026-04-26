"""
Multimodal CNN Model for Skin Cancer Detection
Combines image features (CNN) with metadata features (age, sex, location)
"""

import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights


class MultimodalModel(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()

        self.cnn = resnet50(weights=ResNet50_Weights.DEFAULT)
        in_features = self.cnn.fc.in_features
        self.cnn.fc = nn.Identity()

        self.fc = nn.Sequential(
            nn.Linear(in_features + 2, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, images, ages, genders):
        features = self.cnn(images)
        ages = ages.view(-1, 1)
        genders = genders.view(-1, 1)
        combined = torch.cat([features, ages, genders], dim=1)
        return self.fc(combined)


class ImageOnlyModel(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()
        self.cnn = resnet50(weights=ResNet50_Weights.DEFAULT)
        in_features = self.cnn.fc.in_features
        self.cnn.fc = nn.Linear(in_features, num_classes)

    def forward(self, images, ages=None, genders=None):
        return self.cnn(images)


def get_model(model_type='multimodal', num_classes=7, metadata_dim=2,
              backbone='resnet50', pretrained=True):
    """
    Factory function to get model
    
    Args:
        model_type (str): 'multimodal' or 'image_only'
        num_classes (int): Number of output classes
        metadata_dim (int): Dimension of metadata
        backbone (str): CNN backbone
        pretrained (bool): Use pretrained weights
    
    Returns:
        nn.Module: Model instance
    """
    if model_type == 'image_only':
        return ImageOnlyModel(num_classes)
    return MultimodalModel(num_classes)
