"""
Custom Dataset class for HAM10000 Skin Cancer Dataset
Binary classification: Melanoma vs Others
"""

import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms


class SkinCancerDataset(Dataset):
    """
    Custom Dataset for loading HAM10000 images and metadata.
    Binary classification: melanoma = 1, others = 0
    """
    
    def __init__(self, data_dir, metadata_file, transform=None, mode='train'):
        """
        Args:
            data_dir (str): Directory with all the images
            metadata_file (str): Path to the metadata CSV file
            transform (callable, optional): Optional transform to be applied on images
            mode (str): 'train', 'val', or 'test'
        """
        if isinstance(data_dir, (list, tuple)):
            self.data_dirs = list(data_dir)
        else:
            self.data_dirs = [data_dir]
        # Ignore commented rows in placeholder metadata files
        self.metadata = pd.read_csv(metadata_file, comment="#")
        self.transform = transform
        self.mode = mode

        required_columns = {"image_id", "dx", "age", "sex"}
        missing = required_columns.difference(self.metadata.columns)
        if missing:
            raise ValueError(
                f"metadata_file is missing required columns: {sorted(missing)}"
            )

        if len(self.metadata) == 0:
            raise ValueError(
                "metadata_file has no data rows. Please provide the HAM10000 metadata."
            )
        
        # Binary label encoding: melanoma = 1, others = 0
        self.label_mapping = {
            'mel': 1,    # Melanoma
            'akiec': 0,  # Actinic keratoses
            'bcc': 0,    # Basal cell carcinoma
            'bkl': 0,    # Benign keratosis
            'df': 0,     # Dermatofibroma
            'nv': 0,     # Melanocytic nevi
            'vasc': 0    # Vascular lesions
        }
        
        # Gender encoding: male=0, female=1
        self.sex_mapping = {'male': 0, 'female': 1}
        
        # Calculate mean age for handling missing values
        # Coerce age to numeric for consistent math; keep NaN for missing values
        self.metadata["age"] = pd.to_numeric(self.metadata["age"], errors="coerce")
        self.mean_age = self.metadata["age"].mean()
        
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        # Get image name and path
        image_id = self.metadata.iloc[idx]["image_id"]
        if pd.isna(image_id):
            raise ValueError(f"Missing image_id at index {idx} in metadata_file")
        img_name = str(image_id) + ".jpg"
        img_path = None
        for data_dir in self.data_dirs:
            candidate_path = os.path.join(data_dir, img_name)
            if os.path.exists(candidate_path):
                img_path = candidate_path
                break

        if img_path is None:
            raise FileNotFoundError(
                f"Image not found in any data_dir for image_id '{image_id}'"
            )
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        # Get binary label: melanoma = 1, others = 0
        diagnosis = self.metadata.iloc[idx]['dx']
        label = self.label_mapping[diagnosis]
        
        # Get age - fill missing values with mean age
        age = self.metadata.iloc[idx]['age']
        if pd.isna(age):
            age = self.mean_age
        
        # Normalize age by dividing by 100
        age_normalized = age / 100.0
        
        # Get gender - encode as male=0, female=1
        sex = self.metadata.iloc[idx]['sex']
        if pd.isna(sex) or sex not in self.sex_mapping:
            # Default to male (0) for missing/unknown values
            sex_encoded = 0.0
        else:
            sex_encoded = float(self.sex_mapping[sex])
        
        # Return as separate tensors
        return (
            image,                                      # Image tensor
            torch.tensor(age_normalized, dtype=torch.float32),  # Age tensor
            torch.tensor(sex_encoded, dtype=torch.float32),     # Gender tensor
            torch.tensor(label, dtype=torch.long)              # Label tensor
        )


def get_transforms(mode='train'):
    """
    Get data augmentation transforms for training and validation
    
    Args:
        mode (str): 'train' or 'val'
    
    Returns:
        transforms.Compose: Composed transforms
    """
    if mode == 'train':
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
