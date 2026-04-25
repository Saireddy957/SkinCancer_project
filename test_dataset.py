"""
Test script for the updated SkinCancerDataset
Demonstrates how to use the dataset with the new return format
"""

import sys
import os
sys.path.append('src')

from dataset import SkinCancerDataset, get_transforms
import torch
from torch.utils.data import DataLoader


def test_dataset():
    """
    Test the dataset to verify it works correctly
    """
    print("="*60)
    print("Testing SkinCancerDataset")
    print("="*60)
    
    # Configuration
    colab_metadata = "/content/HAM10000_metadata.csv"
    colab_images = [
        "/content/ham10000_images_part_1",
        "/content/ham10000_images_part_2",
    ]

    if os.path.exists(colab_metadata):
        metadata_file = colab_metadata
        data_dir = colab_images
    else:
        data_dir = "data/images"
        metadata_file = "data/metadata.csv"
    
    # Check if files exist
    if not os.path.exists(metadata_file):
        print(f"Error: {metadata_file} not found!")
        print("Please ensure HAM10000 metadata is in the correct location.")
        return
    
    # Create dataset
    print("\nCreating dataset...")
    transform = get_transforms(mode='train')
    dataset = SkinCancerDataset(
        data_dir=data_dir,
        metadata_file=metadata_file,
        transform=transform,
        mode='train'
    )
    
    print(f"✓ Dataset created successfully!")
    print(f"  Total samples: {len(dataset)}")
    print(f"  Mean age: {dataset.mean_age:.2f} years")
    
    # Test single sample
    print("\n" + "-"*60)
    print("Testing single sample retrieval...")
    print("-"*60)
    
    try:
        # Get first sample
        image, age, gender, label = dataset[0]
        
        print(f"\nSample 0:")
        print(f"  Image shape: {image.shape}")
        print(f"  Image type: {image.dtype}")
        print(f"  Age: {age.item():.4f} (normalized)")
        print(f"  Age (years): {age.item() * 100:.2f}")
        print(f"  Gender: {gender.item():.0f} ({'Male' if gender.item() == 0 else 'Female'})")
        print(f"  Label: {label.item()} ({'Melanoma' if label.item() == 1 else 'Other'})")
        
        print("\n✓ Sample retrieval successful!")
        
    except Exception as e:
        print(f"\n✗ Error retrieving sample: {e}")
        return
    
    # Test DataLoader
    print("\n" + "-"*60)
    print("Testing DataLoader...")
    print("-"*60)
    
    try:
        dataloader = DataLoader(
            dataset,
            batch_size=4,
            shuffle=True,
            num_workers=0  # Use 0 for Windows compatibility
        )
        
        # Get one batch
        images, ages, genders, labels = next(iter(dataloader))
        
        print(f"\nBatch retrieved:")
        print(f"  Images batch shape: {images.shape}")
        print(f"  Ages batch shape: {ages.shape}")
        print(f"  Genders batch shape: {genders.shape}")
        print(f"  Labels batch shape: {labels.shape}")
        
        print(f"\n  Batch details:")
        for i in range(len(labels)):
            print(f"    Sample {i}: Age={ages[i].item()*100:.1f}y, "
                  f"Gender={'M' if genders[i].item()==0 else 'F'}, "
                  f"Label={'Melanoma' if labels[i].item()==1 else 'Other'}")
        
        print("\n✓ DataLoader test successful!")
        
        # Label distribution
        print("\n" + "-"*60)
        print("Checking label distribution...")
        print("-"*60)
        
        melanoma_count = 0
        other_count = 0

        for i in range(len(dataset)):
            _, _, _, label = dataset[i]
            if label.item() == 1:
                melanoma_count += 1
            else:
                other_count += 1
        
        print(f"\nAll samples:")
        print(f"  Melanoma: {melanoma_count}")
        print(f"  Others: {other_count}")
        print(f"  Ratio: {melanoma_count/(melanoma_count+other_count)*100:.1f}% melanoma")

        print("\nDx distribution from metadata:")
        dx_counts = dataset.metadata["dx"].astype(str).str.strip().str.lower().value_counts()
        for dx, count in dx_counts.items():
            print(f"  {dx}: {count}")
        
    except Exception as e:
        print(f"\n✗ Error with DataLoader: {e}")
        return
    
    print("\n" + "="*60)
    print("All tests passed! ✓")
    print("="*60)
    
    print("\nUsage example:")
    print("""
    from dataset import SkinCancerDataset, get_transforms
    from torch.utils.data import DataLoader
    
    # Create dataset
    transform = get_transforms(mode='train')
    dataset = SkinCancerDataset('data/images', 'data/metadata.csv', transform)
    
    # Create dataloader
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Iterate
    for images, ages, genders, labels in loader:
        # images: [batch_size, 3, 224, 224]
        # ages: [batch_size] - normalized (0-1)
        # genders: [batch_size] - 0=male, 1=female
        # labels: [batch_size] - 0=other, 1=melanoma
        pass
    """)


if __name__ == "__main__":
    test_dataset()
