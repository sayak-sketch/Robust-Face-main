from albumentations import (
    Compose, Resize, HorizontalFlip, RandomBrightnessContrast,
    Blur, GaussNoise
)
from albumentations.pytorch import ToTensorV2

def get_train_transforms():
    return Compose([
        Resize(224, 224),
        HorizontalFlip(p=0.5),
        RandomBrightnessContrast(p=0.5),
        Blur(blur_limit=3, p=0.3),
        GaussNoise(var_limit=(10.0, 50.0), p=0.3),  # type: ignore
        ToTensorV2()
    ])  # type: ignore

def get_val_transforms():
    return Compose([
        Resize(224, 224),
        ToTensorV2()
    ])  # type: ignore
