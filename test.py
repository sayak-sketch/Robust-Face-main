import sys
import torch
import pandas as pd
from src.dataset import FaceDataset
from src.augmentations import get_val_transforms
from src.model import DualHeadFaceNet
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def evaluate(model, dataloader, device):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for x, gender in dataloader:
            x = x.to(device)
            out, _ = model(x)
            pred = (torch.sigmoid(out).cpu().numpy() > 0.5).astype(int)
            y_pred.extend(pred.flatten())
            y_true.extend(gender.numpy().astype(int))
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print(f"F1-score: {f1:.4f}")

if __name__ == "__main__":
    test_csv = sys.argv[1]  # e.g., "val.csv"
    model_path = "model.pt"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df = pd.read_csv(test_csv)
    dataset = FaceDataset(df['image_path'].tolist(), df['gender'].tolist(), None, transform=get_val_transforms())
    loader = DataLoader(dataset, batch_size=32, shuffle=False)
    model = DualHeadFaceNet(num_id_classes=2)
    model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
    model.to(device)
    evaluate(model, loader, device)