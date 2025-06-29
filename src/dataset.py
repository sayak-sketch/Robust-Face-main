from torch.utils.data import Dataset
from PIL import Image
import numpy as np

class FaceDataset(Dataset):
    def __init__(self, image_paths, genders, ids, transform=None):
        self.image_paths = image_paths
        self.genders = genders
        self.ids = ids
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        img = np.array(img).astype(np.uint8) 

        if self.transform:
            augmented = self.transform(image=img)
            img = augmented["image"] 

        if self.ids is not None:
            return img, float(self.genders[idx]), self.ids[idx]
        else:
            return img, float(self.genders[idx])
