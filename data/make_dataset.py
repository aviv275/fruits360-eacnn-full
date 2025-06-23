import os
import shutil
import random
from pathlib import Path
from typing import Tuple

DATA_RAW = Path('data/raw/fruits-360')
DATA_PROCESSED = Path('data/processed/fruits-360')
SPLIT = (0.8, 0.2)  # train, val


def check_integrity(data_dir: Path) -> bool:
    train_dir = data_dir / 'train'
    test_dir = data_dir / 'test'
    if not train_dir.exists() or not test_dir.exists():
        print(f"Missing train/ or test/ in {data_dir}")
        return False
    train_classes = set(os.listdir(train_dir))
    test_classes = set(os.listdir(test_dir))
    if not train_classes or not test_classes:
        print("No classes found in train/ or test/")
        return False
    if train_classes != test_classes:
        print("Class mismatch between train/ and test/")
        return False
    print(f"Integrity check passed: {len(train_classes)} classes.")
    return True


def split_train_val(raw_train_dir: Path, processed_dir: Path, split: Tuple[float, float] = SPLIT):
    processed_train = processed_dir / 'train'
    processed_val = processed_dir / 'val'
    processed_test = processed_dir / 'test'
    for d in [processed_train, processed_val, processed_test]:
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True, exist_ok=True)
    classes = os.listdir(raw_train_dir)
    for cls in classes:
        imgs = os.listdir(raw_train_dir / cls)
        random.shuffle(imgs)
        n_train = int(len(imgs) * split[0])
        train_imgs = imgs[:n_train]
        val_imgs = imgs[n_train:]
        (processed_train / cls).mkdir(parents=True, exist_ok=True)
        (processed_val / cls).mkdir(parents=True, exist_ok=True)
        for img in train_imgs:
            shutil.copy(raw_train_dir / cls / img, processed_train / cls / img)
        for img in val_imgs:
            shutil.copy(raw_train_dir / cls / img, processed_val / cls / img)
    print(f"Split complete: {len(classes)} classes.")

    # Copy test set as is
    raw_test_dir = DATA_RAW / 'test'
    for cls in os.listdir(raw_test_dir):
        (processed_test / cls).mkdir(parents=True, exist_ok=True)
        for img in os.listdir(raw_test_dir / cls):
            shutil.copy(raw_test_dir / cls / img, processed_test / cls / img)

    # Update config.py
    config_path = Path('src/config.py')
    with open(config_path, 'r') as f:
        lines = f.readlines()
    with open(config_path, 'w') as f:
        for line in lines:
            if line.startswith('DATA_DIR'):
                f.write("DATA_DIR = 'data/processed/fruits-360/'\n")
            else:
                f.write(line)
    print("src/config.py updated to use processed dataset.")


def main():
    if not check_integrity(DATA_RAW):
        return
    split_train_val(DATA_RAW / 'train', DATA_PROCESSED)
    print("Done.")

if __name__ == '__main__':
    main() 