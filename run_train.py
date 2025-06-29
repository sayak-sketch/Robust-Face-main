import numpy as np
import torch
import pandas as pd
from torch.utils.data import DataLoader
from src.dataset import FaceDataset
from src.augmentations import get_train_transforms, get_val_transforms
from src.train import train_model
from PIL import Image
import os

train_df = pd.read_csv("train.csv")
val_df = pd.read_csv("val.csv")

train_dataset = FaceDataset(
    train_df['image_path'].tolist(),
    train_df['gender'].tolist(),
    None,  # No identity for gender-only
    transform=get_train_transforms()
)
val_dataset = FaceDataset(
    val_df['image_path'].tolist(),
    val_df['gender'].tolist(),
    None,
    transform=get_val_transforms()
)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_model(train_loader, val_loader, num_classes=2, device=device)

def match_face(distorted_img_path, identities_dir, model, transform, threshold=0.7):
    distorted_img = Image.open(distorted_img_path).convert("RGB")
    distorted_tensor = transform(image=np.array(distorted_img))["image"].unsqueeze(0)
    with torch.no_grad():
        distorted_emb = model.get_embedding(distorted_tensor)

    for identity in os.listdir(identities_dir):
        id_folder = os.path.join(identities_dir, identity)
        for ref_img_name in os.listdir(id_folder):
            ref_img = Image.open(os.path.join(id_folder, ref_img_name)).convert("RGB")
            ref_tensor = transform(image=np.array(ref_img))["image"].unsqueeze(0)
            with torch.no_grad():
                ref_emb = model.get_embedding(ref_tensor)
            dist = torch.norm(distorted_emb - ref_emb).item()
            if dist < threshold:
                return identity, 1  # Positive match
    return None, 0  # Negative match
