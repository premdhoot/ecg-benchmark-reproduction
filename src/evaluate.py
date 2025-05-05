# evaluate.py
import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, f1_score
from models.resnet1d import resnet18_1d

class ECGDataset(Dataset):
    def __init__(self, signal_path, label_path):
        self.signals = np.load(signal_path)
        self.labels = np.load(label_path)

    def __len__(self):
        return len(self.signals)

    def __getitem__(self, idx):
        x = torch.tensor(self.signals[idx], dtype=torch.float32)
        y = torch.tensor(self.labels[idx], dtype=torch.float32)
        return x, y

def evaluate(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            preds = torch.sigmoid(logits).cpu().numpy()
            all_preds.append(preds)
            all_labels.append(y.cpu().numpy())

    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)

    roc_auc = roc_auc_score(all_labels, all_preds, average="macro")
    pred_binary = (all_preds >= 0.5).astype(int)
    f1 = f1_score(all_labels, pred_binary, average="macro")

    return roc_auc, f1

def main():
    signal_path = "data/test_signals.npy"
    label_path = "data/test_labels.npy"
    checkpoint_path = "models/resnet_epoch5.pth"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = ECGDataset(signal_path, label_path)
    loader = DataLoader(dataset, batch_size=64)

    num_classes = dataset.labels.shape[1]
    model = resnet18_1d(num_classes).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    roc_auc, f1 = evaluate(model, loader, device)
    print(f"Test ROC-AUC: {roc_auc:.4f}")
    print(f"Test F1 Score: {f1:.4f}")

if __name__ == "__main__":
    main()
