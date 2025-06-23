#!/usr/bin/env python3
"""
Create a diverse fruits-360-mini dataset with different types of fruits
"""

import os
import shutil
from glob import glob

SRC_DIR = 'data/raw/fruits-360'
DST_DIR = 'data/raw/fruits-360-diverse'
N_TRAIN = 100
N_TEST = 20

# Select diverse fruits (not just apples)
DIVERSE_FRUITS = [
    'Apple Red 1',           # Apple
    'Banana 4',              # Banana
    'Orange 1',              # Orange (if exists, otherwise use Clementine)
    'Pear 1',                # Pear
    'Strawberry 1',          # Strawberry (if exists, otherwise use Strawberry Wedge 1)
    'Pineapple 1',           # Pineapple
    'Kiwi 1',                # Kiwi
    'Grape 1',               # Grape (if exists, otherwise use another fruit)
    'Lemon 1',               # Lemon (if exists, otherwise use Lemon Meyer 1)
    'Peach 1'                # Peach (if exists, otherwise use Peach Flat 1)
]

# Fallback fruits if some don't exist
FALLBACK_FRUITS = {
    'Orange 1': 'Clementine 1',
    'Strawberry 1': 'Strawberry Wedge 1',
    'Grape 1': 'Grapefruit Pink 1',
    'Lemon 1': 'Lemon Meyer 1',
    'Peach 1': 'Peach Flat 1'
}

def get_available_fruits():
    """Get list of available fruits from the source directory"""
    train_dir = os.path.join(SRC_DIR, 'train')
    return sorted([d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))])

def select_fruits():
    """Select fruits for the dataset, using fallbacks if needed"""
    available_fruits = get_available_fruits()
    selected_fruits = []
    
    for fruit in DIVERSE_FRUITS:
        if fruit in available_fruits:
            selected_fruits.append(fruit)
        elif fruit in FALLBACK_FRUITS and FALLBACK_FRUITS[fruit] in available_fruits:
            selected_fruits.append(FALLBACK_FRUITS[fruit])
            print(f"Using fallback: {FALLBACK_FRUITS[fruit]} instead of {fruit}")
        else:
            # Find a similar fruit
            for available in available_fruits:
                if fruit.split()[0].lower() in available.lower() and available not in selected_fruits:
                    selected_fruits.append(available)
                    print(f"Using alternative: {available} instead of {fruit}")
                    break
    
    # If we still don't have enough, add some more diverse fruits
    while len(selected_fruits) < 10:
        for available in available_fruits:
            if available not in selected_fruits and len(selected_fruits) < 10:
                selected_fruits.append(available)
                print(f"Added: {available}")
                break
    
    return selected_fruits[:10]  # Ensure we have exactly 10

def create_dataset():
    """Create the diverse dataset"""
    selected_fruits = select_fruits()
    
    print(f"Selected fruits: {selected_fruits}")
    
    for split in ['train', 'test']:
        src_split = os.path.join(SRC_DIR, split)
        dst_split = os.path.join(DST_DIR, split)
        os.makedirs(dst_split, exist_ok=True)
        
        for cls in selected_fruits:
            src_cls = os.path.join(src_split, cls)
            dst_cls = os.path.join(dst_split, cls)
            
            if not os.path.exists(src_cls):
                print(f"Warning: {cls} not found in {split}")
                continue
                
            os.makedirs(dst_cls, exist_ok=True)
            images = sorted(glob(os.path.join(src_cls, '*.jpg')))
            n = N_TRAIN if split == 'train' else N_TEST
            
            copied = 0
            for img_path in images[:n]:
                shutil.copy(img_path, dst_cls)
                copied += 1
                
            print(f"Copied {copied} images for class {cls} in {split}")
    
    print(f"\nDone! Created {DST_DIR} with {len(selected_fruits)} diverse fruits:")
    for i, fruit in enumerate(selected_fruits, 1):
        print(f"  {i}. {fruit}")

if __name__ == '__main__':
    create_dataset() 