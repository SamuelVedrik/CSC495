# ======== IMPORTS =======
import random
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import ImageOps, Image
from constants import RESIZE, NORMALIZE

class FullDataset(Dataset):
    
    def __init__(self, csv, transform=None, parent_path="data/"):
        self.csv = csv
        self.id_to_labels = {i: v for i, v in enumerate(self.csv["Label"].unique())}
        self.labels_to_id = {v: i for i, v in self.id_to_labels.items()}
        self.transform = transform
        self.parent_path = parent_path
        
    def __getitem__(self, idx):
        
        path, label = self.csv.iloc[idx]
        image = Image.open(os.path.join(self.parent_path, path)).convert("RGB") # Always RGB
        if self.transform: 
            image = self.transform(image)
        return image, self.labels_to_id[label]

    def __len__(self):
        return self.csv.shape[0]
    
    def split_dataset(self, num_per_class, seed=None):
        full = [self.csv[self.csv["Label"] == label].sample(num_per_class, random_state=seed) for label in self.id_to_labels.values()]
        full = pd.concat(full)
        return FullDataset(full, self.transform, self.parent_path)
    
def get_transformations():
    
    def _gaussian_noise(x):
        return x + (0.05 **0.5) * torch.randn(*x.shape)

    def _sp_noise(x): 
        x = x.clone()
        rand1 = torch.rand(*x.shape)
        rand2 = torch.rand(*x.shape)
        x = torch.where(rand1 < 0.1, torch.FloatTensor([0.]), x)
        x = torch.where(rand2 < 0.1, torch.FloatTensor([1.]), x)
        return x
    
    return dict(
        standard = transforms.Compose([
            transforms.ToTensor(),
            RESIZE, 
            NORMALIZE
        ]),
        random_erase = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomErasing(p=1),
            RESIZE, 
            NORMALIZE
        ]),
        gaussian_noise = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(_gaussian_noise),
            RESIZE,
            NORMALIZE
        ]),
        salt_pepper_noise = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(_sp_noise),
            RESIZE,
            NORMALIZE
        ]),
        blurred = transforms.Compose([
            transforms.ToTensor(),
            transforms.GaussianBlur(7),
            RESIZE,
            NORMALIZE
        ]),
        equalized = transforms.Compose([
            transforms.Lambda(ImageOps.equalize),
            transforms.ToTensor(),
            RESIZE,
            NORMALIZE
        ]),
        inverted = transforms.Compose([
            transforms.Lambda(ImageOps.invert),
            transforms.ToTensor(),
            RESIZE,
            NORMALIZE
        ]),
        solarized = transforms.Compose([
            transforms.Lambda(ImageOps.solarize),
            transforms.ToTensor(),
            RESIZE,
            NORMALIZE
        ]),
        perspective = transforms.Compose([
            transforms.ToTensor(),
            RESIZE,
            transforms.RandomPerspective(p=1),
            NORMALIZE
        ]),
        hori_flip = transforms.Compose([
            transforms.ToTensor(), 
            RESIZE, 
            transforms.RandomHorizontalFlip(p=1),
            NORMALIZE
        ]),
        rotate_90 = transforms.Compose([
            transforms.ToTensor(),
            RESIZE, 
            transforms.RandomRotation((90, 90)),
            NORMALIZE
        ]),
        grayscale = transforms.Compose([
            transforms.ToTensor(),
            RESIZE,
            transforms.Grayscale(num_output_channels=3),
            NORMALIZE
        ])
    )
    
    
    
    
        
    
    

    
    
    
    
    