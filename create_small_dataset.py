#!/usr/bin/env python3
"""
Script to create a small dataset from Fruits360 with 10 diverse fruits for fast training.
"""

import os
import shutil
import random
from pathlib import Path

def create_small_dataset():
    # Define 10 diverse fruits to sample
    selected_fruits = [
        "Apple Golden 1",      # Golden apple
        "Banana 1",            # Banana
        "Orange 1",            # Orange
        "Strawberry 1",        # Strawberry
        "Grape White 1",       # White grape
        "Lemon 1",             # Lemon
        "Mango 1",             # Mango
        "Pineapple 1",         # Pineapple
        "Tomato 1",            # Tomato
        "Kiwi 1"               # Kiwi
    ]
    
    # Source and destination paths
    source_train = Path("data/raw/fruits-360/Training")
    source_test = Path("data/raw/fruits-360/Test")
    dest_train = Path("data/small_dataset/Training")
    dest_test = Path("data/small_dataset/Test")
    
    # Create destination directories
    dest_train.mkdir(parents=True, exist_ok=True)
    dest_test.mkdir(parents=True, exist_ok=True)
    
    print("Creating small dataset with 10 fruits...")
    print("Selected fruits:", selected_fruits)
    
    # Copy training data
    for fruit in selected_fruits:
        print(f"Processing {fruit}...")
        
        # Copy training data
        src_train = source_train / fruit
        dst_train = dest_train / fruit
        
        if src_train.exists():
            shutil.copytree(src_train, dst_train, dirs_exist_ok=True)
            print(f"  ✓ Copied training data for {fruit}")
        else:
            print(f"  ✗ Training data not found for {fruit}")
        
        # Copy test data
        src_test = source_test / fruit
        dst_test = dest_test / fruit
        
        if src_test.exists():
            shutil.copytree(src_test, dst_test, dirs_exist_ok=True)
            print(f"  ✓ Copied test data for {fruit}")
        else:
            print(f"  ✗ Test data not found for {fruit}")
    
    # Print dataset statistics
    print("\n" + "="*50)
    print("DATASET STATISTICS")
    print("="*50)
    
    total_train_images = 0
    total_test_images = 0
    
    for fruit in selected_fruits:
        train_path = dest_train / fruit
        test_path = dest_test / fruit
        
        if train_path.exists():
            train_count = len(list(train_path.glob("*.jpg"))) + len(list(train_path.glob("*.png")))
            total_train_images += train_count
            print(f"{fruit}: {train_count} training images")
        
        if test_path.exists():
            test_count = len(list(test_path.glob("*.jpg"))) + len(list(test_path.glob("*.png")))
            total_test_images += test_count
            print(f"{fruit}: {test_count} test images")
    
    print(f"\nTotal training images: {total_train_images}")
    print(f"Total test images: {total_test_images}")
    print(f"Total images: {total_train_images + total_test_images}")
    print(f"Number of classes: {len(selected_fruits)}")
    
    # Create a class mapping file
    class_mapping = {i: fruit for i, fruit in enumerate(selected_fruits)}
    
    import json
    with open("data/small_dataset/class_mapping.json", "w") as f:
        json.dump(class_mapping, f, indent=2)
    
    print(f"\nClass mapping saved to: data/small_dataset/class_mapping.json")
    print(f"Small dataset created at: data/small_dataset/")
    
    return selected_fruits, total_train_images, total_test_images

if __name__ == "__main__":
    create_small_dataset() 