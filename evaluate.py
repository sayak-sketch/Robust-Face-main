import sys
import torch
import pandas as pd
import numpy as np
from src.dataset import FaceDataset
from src.augmentations import get_val_transforms
from src.model import DualHeadFaceNet
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics.pairwise import cosine_similarity

# Helper: extract embeddings for all images in a DataLoader
def extract_embeddings(model, dataloader, device):
    model.eval()
    embeddings = []
    labels = []
    with torch.no_grad():
        for x, id_label in dataloader:
            x = x.to(device)
            _, emb = model(x)
            embeddings.append(emb.cpu().numpy())
            labels.extend(id_label.numpy())
    embeddings = np.concatenate(embeddings, axis=0)
    return embeddings, np.array(labels)

# Helper: compute pairwise identity matches and metrics
def evaluate_verification(embeddings, labels, threshold=0.5):
    # Compute cosine similarity matrix
    sim_matrix = cosine_similarity(embeddings)
    n = len(labels)
    y_true, y_pred = [], []
    for i in range(n):
        for j in range(i+1, n):
            same = int(labels[i] == labels[j])
            score = sim_matrix[i, j]
            match = int(score > threshold)
            y_true.append(same)
            y_pred.append(match)
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print(f"F1-score: {f1:.4f}")
    return acc, prec, rec, f1

if __name__ == "__main__":
    csv_path = sys.argv[1]  # e.g., "identities_val.csv"
    model_path = "model.pt"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df = pd.read_csv(csv_path)
    # Expect columns: image_path, id (or identity)
    id_col = 'id' if 'id' in df.columns else 'identity'
    dataset = FaceDataset(df['image_path'].tolist(), df[id_col].tolist(), None, transform=get_val_transforms(), task='B')
    loader = DataLoader(dataset, batch_size=32, shuffle=False)
    model = DualHeadFaceNet(num_id_classes=2)
    model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
    model.to(device)
    embeddings, labels = extract_embeddings(model, loader, device)
    # You may want to tune the threshold for your use case
    evaluate_verification(embeddings, labels, threshold=0.5)