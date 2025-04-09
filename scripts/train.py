import torch
import torch.nn as nn
import torch.optim as optim
from efficientnet_pytorch import EfficientNet
from dataset import FaceTensorDataset
from torch.utils.data import DataLoader, random_split
import os
from tqdm import tqdm
import matplotlib.pyplot as plt

# Configuration
root_dir = "extracted_faces"
num_frames = 5
batch_size = 8
num_epochs = 5
model_save_path = "models/efficientnet_faces.pth"

# Dataset and split into train/val
full_dataset = FaceTensorDataset(root_dir, num_frames=num_frames, oversample=True)
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Model
model = EfficientNet.from_pretrained('efficientnet-b0')
model._fc = nn.Linear(model._fc.in_features, 2)
model.to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Tracking
train_accuracies = []
val_accuracies = []
train_losses = []
val_losses = []


# Training loop
for epoch in range(num_epochs):
    model.train()
    total, correct, loss_total = 0, 0, 0

    for faces, labels, _ in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
        B, N, C, H, W = faces.size()
        faces = faces.view(B * N, C, H, W).to(device)
        labels_exp = labels.unsqueeze(1).expand(-1, N).reshape(-1).to(device)

        optimizer.zero_grad()
        outputs = model(faces)
        loss = criterion(outputs, labels_exp)
        loss.backward()
        optimizer.step()

        loss_total += loss.item()
        _, pred = torch.max(outputs, 1)
        correct += (pred == labels_exp).sum().item()
        total += labels_exp.size(0)

    train_acc = 100 * correct / total
    train_losses.append(loss_total/ len(train_loader))
    train_accuracies.append(train_acc)

    # 🔍 Validation
    model.eval()
    val_correct, val_total, val_loss_total = 0, 0, 0
    with torch.no_grad():
        for faces, labels, _ in val_loader:
            B, N, C, H, W = faces.size()
            faces = faces.view(B * N, C, H, W).to(device)
            labels_exp = labels.unsqueeze(1).expand(-1, N).reshape(-1).to(device)
            outputs = model(faces)
            loss = criterion(outputs, labels_exp)

            val_loss_total += loss.item()
            _, pred = torch.max(outputs, 1)
            val_correct += (pred == labels_exp).sum().item()
            val_total += labels_exp.size(0)

    val_acc = 100 * val_correct / val_total
    val_losses.append(val_loss_total/ len(val_loader))
    val_accuracies.append(val_acc)

    avg_train_loss = loss_total / len(train_loader)
    avg_val_loss = val_loss_total / len(val_loader)

    print(f"Epoch {epoch+1} - Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}% | Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}%")

# Save model
os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
torch.save(model.state_dict(), model_save_path)
print(f"\n✅ Model saved to {model_save_path}")

# 📊 Plotting
plt.figure(figsize=(10, 4))

# Accuracy plot
plt.subplot(1, 2, 1)
plt.plot(train_accuracies, label='Train Acc')
plt.plot(val_accuracies, label='Val Acc')
plt.title("Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.legend()

# Loss plot
plt.subplot(1, 2, 2)
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.title("Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.tight_layout()
plt.savefig("training_metrics.png")
plt.show()
