import os
import requests
import torch
import numpy as np
import pandas as pd
import torchvision.transforms as transforms
from PIL import Image
from io import BytesIO
from tqdm import tqdm, trange
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader


class FaceFeatureDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.from_numpy(features).float()
        self.labels = torch.from_numpy(labels).float()

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        feature = self.features[idx]
        label = self.labels[idx].unsqueeze(-1) # shape(1,)
        return feature, label


class FaceDataset(Dataset):
    def __init__(self, imgs, labels, img_size=(112, 112)):
        super().__init__()
        self.imgs = imgs
        self.labels = labels
        self.img_size = img_size
        self.transform = transforms.Compose([
            transforms.Resize(self.img_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],      
                std=[0.229, 0.224, 0.225]
            )
        ])

    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, idx):
        label = self.labels[idx]
        img_path = self.imgs[idx]
        try:
            img = Image.open(img_path).convert('RGB')
        except Exception:
            print(f"Bad image: {img_path}")
            img = Image.new('RGB', self.img_size, color='black')
            label = 0.0
        
        img = self.transform(img)
        return img, label


class FaceInferenceDataset(Dataset):
    """用于图像推理的 Dataset"""
    def __init__(self, imgs, img_size=(112, 112)):
        self.imgs = imgs
        self.img_size = img_size
        self.transform = transforms.Compose([
            transforms.Resize(self.img_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],      
                std=[0.229, 0.224, 0.225]
            )
        ])

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        path = self.imgs[idx]
        try:
            img = Image.open(path).convert('RGB')
            img_tensor = self.transform(img)
            return img_tensor, path
        except Exception as e:
            print(f"Warning: Could not load image {path}, skipping. Error: {e}")
            return torch.zeros(3, 112, 112), path


if __name__ == "__main__":
    data = pd.read_parquet("dataset/xxx_v0_data.parquet")
    imgs = data['fpath'].values
    labels = data['cr_label'].values
    X_train, X_val, y_train, y_val = train_test_split(imgs, labels, test_size=0.05, shuffle=True, random_state=42)
    train_dataset = FaceDataset(X_train, y_train, img_size=(112, 112))
    val_dataset = FaceDataset(X_val, y_val, img_size=(112, 112))

    train_loader = DataLoader(train_dataset, batch_size=256, num_workers=0, pin_memory=True, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=256, num_workers=0, pin_memory=True, shuffle=False, drop_last=False)

    for batch_idx, (imgs, labels) in enumerate(tqdm(train_loader, desc="处理批次")):
        print(f"批次 {batch_idx}: imgs {imgs.shape}, labels {labels.shape}")