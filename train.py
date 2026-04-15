import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

df = pd.read_csv("data/landmarks.csv", header=None)
df.columns = ["label"] + [f"f{i}" for i in range(63)]

le = LabelEncoder()
y  = le.fit_transform(df["label"].values)
X  = df.drop("label", axis=1).values.astype(np.float32)

os.makedirs("model", exist_ok=True)
np.save("model/label_map.npy", le.classes_)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
num_classes = len(le.classes_)
print(f"Classes ({num_classes}): {list(le.classes_)}")
print(f"Train: {len(X_train)} | Test: {len(X_test)}\n")

train_loader = DataLoader(TensorDataset(torch.tensor(X_train), torch.tensor(y_train)), batch_size=32, shuffle=True)
test_loader  = DataLoader(TensorDataset(torch.tensor(X_test),  torch.tensor(y_test)),  batch_size=32)

class LandmarkMLP(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(63, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
    def forward(self, x):
        return self.net(x)

model     = LandmarkMLP(num_classes).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2)

best_acc, patience, no_improve = 0, 5, 0
print("--- Training ---")

for epoch in range(100):
    model.train()
    total_loss, correct_train, total_train = 0, 0, 0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        out  = model(X_batch)
        loss = criterion(out, y_batch)
        loss.backward()
        optimizer.step()
        total_loss    += loss.item() * len(y_batch)
        correct_train += (out.argmax(1) == y_batch).sum().item()
        total_train   += len(y_batch)

    model.eval()
    with torch.no_grad():
        correct_val = sum((model(X.to(device)).argmax(1) == y.to(device)).sum().item()
                          for X, y in test_loader)
    train_acc = correct_train / total_train
    val_acc   = correct_val   / len(y_test)
    scheduler.step(1 - val_acc)

    marker = " ✓ best" if val_acc > best_acc else ""
    print(f"Epoch {epoch+1:>3} | loss: {total_loss/total_train:.4f} | train_acc: {train_acc:.4f} | val_acc: {val_acc:.4f}{marker}")

    if val_acc > best_acc:
        best_acc, no_improve = val_acc, 0
        torch.save(model.state_dict(), "model/sign_model.pth")
    else:
        no_improve += 1
        if no_improve >= patience:
            print("Early stopping.")
            break

print(f"\nDone! Best val_acc: {best_acc:.4f} — model saved to model/sign_model.pth")
