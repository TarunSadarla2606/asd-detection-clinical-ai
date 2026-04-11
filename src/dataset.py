"""
dataset.py
----------
PyTorch Dataset class for loading ABIDE-I sMRI PNG slices.

The CSVs contain columns:
    [0] index
    [1] Image_paths     — path to PNG slice
    [3] LABEL           — 0 = Typical Control, 1 = ASD
    [4] Preprocessed_Image_paths  (alternative path column in some CSVs)

Usage:
    from dataset import ASDDataset
    from torch.utils.data import DataLoader

    train_ds = ASDDataset('extracted_random_labels_train.csv', mode='cnn')
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
"""

import torch
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from preprocess import preprocess_cnn, preprocess_vit


class ASDDataset(Dataset):
    """
    Dataset for ASD/TC binary classification from ABIDE-I sMRI slices.

    Args:
        csv_file:     Path to label CSV file.
        mode:         'cnn' or 'vit' — determines preprocessing pipeline.
        path_prefix:  Optional string to replace local Windows paths with
                      the correct runtime path (e.g., Kaggle input path).
        transform:    Optional additional torchvision transforms (applied
                      after the standard preprocessing).

    Label encoding:
        0 = Typical Control (TC)
        1 = Autism Spectrum Disorder (ASD)
    """

    # Windows path used during development; replace at runtime
    _LOCAL_PATH = "E:\\TARUN\\Projects\\Autism Detection\\Data\\data_png"

    def __init__(self, csv_file: str, mode: str = 'cnn',
                 path_prefix: str = '/kaggle/input/autism/',
                 transform=None):
        self.data       = pd.read_csv(csv_file)
        self.mode       = mode.lower()
        self.path_prefix = path_prefix
        self.transform  = transform

        assert self.mode in ('cnn', 'vit'), f"mode must be 'cnn' or 'vit', got {mode}"

        # Detect which column holds image paths
        self._path_col  = 4 if 'Preprocessed_Image_paths' in self.data.columns else 1

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int):
        img_path = self.data.iloc[index, self._path_col]
        img_path = img_path.replace(self._LOCAL_PATH, self.path_prefix).replace("\\", "/")
        label    = int(self.data.iloc[index, 3])

        # Apply preprocessing
        if self.mode == 'cnn':
            img_np = preprocess_cnn(img_path)
        else:
            img_np = preprocess_vit(img_path)

        # HWC → CHW, float32 tensor
        img_tensor = torch.from_numpy(img_np.transpose(2, 0, 1)).float()

        if self.transform:
            img_tensor = self.transform(img_tensor)

        return img_tensor, label


def build_loaders(train_csv: str, val_csv: str, test_csv: str,
                  mode: str = 'cnn',
                  batch_size: int = 64,
                  path_prefix: str = '/kaggle/input/autism/',
                  num_workers: int = 2):
    """
    Build train, validation, and test DataLoaders.

    Args:
        train_csv, val_csv, test_csv: Paths to split CSV files.
        mode:        'cnn' or 'vit'.
        batch_size:  Batch size for all loaders.
        path_prefix: Runtime path prefix for image files.
        num_workers: Number of DataLoader workers.

    Returns:
        (train_loader, val_loader, test_loader)
    """
    train_ds = ASDDataset(train_csv, mode=mode, path_prefix=path_prefix)
    val_ds   = ASDDataset(val_csv,   mode=mode, path_prefix=path_prefix)
    test_ds  = ASDDataset(test_csv,  mode=mode, path_prefix=path_prefix)

    from torch.utils.data import DataLoader
    kw = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=True)
    return (
        DataLoader(train_ds, shuffle=True,  **kw),
        DataLoader(val_ds,   shuffle=False, **kw),
        DataLoader(test_ds,  shuffle=False, **kw),
    )


if __name__ == "__main__":
    # Quick size check without loading images
    import sys
    csv = sys.argv[1] if len(sys.argv) > 1 else "extracted_random_labels_train.csv"
    ds = ASDDataset.__new__(ASDDataset)
    ds.data = pd.read_csv(csv)
    print(f"Dataset size: {len(ds.data):,} slices")
    print(f"ASD: {(ds.data.iloc[:,3]==1).sum():,}  TC: {(ds.data.iloc[:,3]==0).sum():,}")
