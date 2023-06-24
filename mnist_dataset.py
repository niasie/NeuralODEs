import gzip
import numpy as np
import os
import torch
from torch.utils.data import TensorDataset, DataLoader


IMAGE_SIZE = 28


def load_images(device, split="train"):
    data_path = os.path.join("data", "mnist", f'{split}_images.gz')
    f = gzip.open(data_path,"r")
    f.read(16)
    buf = f.read()
    data = np.frombuffer(buf, dtype=np.uint8)
    data = data.reshape(-1, IMAGE_SIZE, IMAGE_SIZE)
    return torch.from_numpy(normalize(data)).to(dtype=torch.float32, device=device)


def load_labels(device, split="train"):
    label_path = os.path.join("data", "mnist", f'{split}_labels.gz')
    f = gzip.open(label_path,"r")    
    f.read(8)
    buf = f.read()
    labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
    return torch.from_numpy(labels).to(dtype=torch.int64, device=device)


def getMNISTTDataset(
        device,
        split="train",
        num_workers=0,
        batch_size=5000,
        shuffle=True,
        ):
    data = load_images(device, split)
    labels = load_labels(device, split)
    return DataLoader(
        TensorDataset(data, labels),
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle
    )


def normalize(x):
    new_x = 2.0 * x / 255.0 - 1.0
    return new_x
