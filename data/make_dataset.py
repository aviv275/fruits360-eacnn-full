#!/usr/bin/env python3
"""
Create a larger fruits-360-mini-bigger dataset with the first 10 classes,
100 train images and 20 test images per class.
"""
import os
import shutil
from glob import glob

SRC_DIR = 'data/raw/fruits-360'
DST_DIR = 'data/raw/fruits-360-mini-bigger'
N_CLASSES = 10
N_TRAIN = 100
N_TEST = 20

for split in ['train', 'test']:
    src_split = os.path.join(SRC_DIR, split)
    dst_split = os.path.join(DST_DIR, split)
    os.makedirs(dst_split, exist_ok=True)
    # Get sorted class names
    class_names = sorted([d for d in os.listdir(src_split) if os.path.isdir(os.path.join(src_split, d))])[:N_CLASSES]
    for cls in class_names:
        src_cls = os.path.join(src_split, cls)
        dst_cls = os.path.join(dst_split, cls)
        os.makedirs(dst_cls, exist_ok=True)
        images = sorted(glob(os.path.join(src_cls, '*.jpg')))
        n = N_TRAIN if split == 'train' else N_TEST
        for img_path in images[:n]:
            shutil.copy(img_path, dst_cls)
        print(f"Copied {min(n, len(images))} images for class {cls} in {split}")

print(f"Done! Created {DST_DIR} with {N_CLASSES} classes, {N_TRAIN} train and {N_TEST} test images per class.") 