"""
Multimodal CNN Model for Skin Cancer Detection
Combines image features (CNN) with metadata features (age, sex, location)
"""

import torch
import torch.nn as nn
import torchvision.models as models


class MultimodalCNN(nn.Module):
    """
    Multimodal CNN that combines:
    - Image features from pretrained CNN (ResNet, EfficientNet, etc.)
    - Metadata features (age, sex, localization)
    """
    
    def __init__(self, num_classes=7, metadata_dim=2, backbone='resnet50', pretrained=True):
        """
        Args:
            num_classes (int): Number of skin lesion classes
            metadata_dim (int): Dimension of metadata features
            backbone (str): CNN backbone architecture
            pretrained (bool): Use pretrained ImageNet weights
        """
        super(MultimodalCNN, self).__init__()
        
        self.backbone_name = backbone
        
        # Load pretrained CNN backbone
        if backbone == 'resnet50':
            self.backbone = models.resnet50(pretrained=pretrained)
            image_feature_dim = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()  # Remove final FC layer
            
        elif backbone == 'efficientnet_b0':
            self.backbone = models.efficientnet_b0(pretrained=pretrained)
            image_feature_dim = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Identity()
            
        elif backbone == 'mobilenet_v2':
            self.backbone = models.mobilenet_v2(pretrained=pretrained)
            image_feature_dim = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Identity()
            
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        # Metadata processing network
        self.metadata_net = nn.Sequential(
            nn.Linear(metadata_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 16),
            nn.ReLU()
        )
        
        # Fusion layer
        combined_dim = image_feature_dim + 16
        
        self.fusion = nn.Sequential(
            nn.Linear(combined_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, image, metadata):
        """
        Forward pass
        
        Args:
            image (torch.Tensor): Batch of images [B, C, H, W]
            metadata (torch.Tensor): Batch of metadata features [B, metadata_dim]
        
        Returns:
            torch.Tensor: Class logits [B, num_classes]
        """
        # Extract image features
        image_features = self.backbone(image)
        
        # Process metadata
        metadata_features = self.metadata_net(metadata)
        
        # Concatenate features
        combined = torch.cat([image_features, metadata_features], dim=1)
        
        # Final classification
        output = self.fusion(combined)
        
        return output


class ImageOnlyCNN(nn.Module):
    """
    Baseline model using only images (no metadata)
    """
    
    def __init__(self, num_classes=7, backbone='resnet50', pretrained=True):
        super(ImageOnlyCNN, self).__init__()
        
        if backbone == 'resnet50':
            self.backbone = models.resnet50(pretrained=pretrained)
            in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Linear(in_features, num_classes)
            
        elif backbone == 'efficientnet_b0':
            self.backbone = models.efficientnet_b0(pretrained=pretrained)
            in_features = self.backbone.classifier[1].in_features
            self.backbone.classifier[1] = nn.Linear(in_features, num_classes)
            
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
    
    def forward(self, image, metadata=None):
        return self.backbone(image)


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
    if model_type == 'multimodal':
        return MultimodalCNN(num_classes, metadata_dim, backbone, pretrained)
    elif model_type == 'image_only':
        return ImageOnlyCNN(num_classes, backbone, pretrained)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
