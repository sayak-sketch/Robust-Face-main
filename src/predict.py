import torch
from PIL import Image
import numpy as np
from src.model import DualHeadFaceNet
from src.augmentations import get_val_transforms

def predict(image_path, model_path):
    model = DualHeadFaceNet(num_id_classes=2) 
    model.load_state_dict(torch.load(model_path, map_location='cpu'), strict=False)
    model.eval()
    
    transform = get_val_transforms()
    img = Image.open(image_path).convert("RGB")
    img_tensor = transform(image=np.array(img))["image"].unsqueeze(0)

    with torch.no_grad():
        gender_logits, _ = model(img_tensor)
        gender = torch.sigmoid(gender_logits).item()
        print(f"Raw gender score: {gender}")

    print(f"Predicted Gender: {'Female' if gender < 0.5 else 'Male'}")

if __name__ == "__main__":
    import sys
    image_path = sys.argv[1]
    model_path = "model.pt"
    predict(image_path, model_path)
