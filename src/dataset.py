from torch.utils.data import Dataset
from PIL import Image
import numpy as np

class FaceDataset(Dataset):
    def _init_(self, image_paths, labels, ids=None, transform=None, task='A'):
        self.image_paths = image_paths
        self.labels = labels  # gender for Task A, identity for Task B
        self.ids = ids
        self.transform = transform
        self.task = task  # 'A' for gender, 'B' for identity

    def _len_(self):
        return len(self.image_paths)

    def _getitem_(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        img = np.array(img).astype(np.uint8) 

        if self.transform:
            augmented = self.transform(image=img)
            img = augmented["image"] 

        if self.task == 'A':
            # Task A: gender classification
            return img, float(self.labels[idx])
        else:
            # Task B: identity recognition (convert string label to int if needed)
            label = self.labels[idx]
            if isinstance(label, str):
                # Map string identities to integer labels for PyTorch
                label = hash(label) % (10 ** 8)  # Simple hash to int
            return img, int(label)
