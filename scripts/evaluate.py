import torch
from torch.utils.data import DataLoader
from dataset import FaceTensorDataset
from efficientnet_pytorch import EfficientNet
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# Set test data directory
test_dir = "extracted_faces_test"
output_dir = "evaluation_output"
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "predictions.txt")

# Dataset and DataLoader
dataset = FaceTensorDataset(test_dir)
dataloader = DataLoader(dataset, batch_size=8)

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
model = EfficientNet.from_pretrained('efficientnet-b0')
model._fc = torch.nn.Linear(model._fc.in_features, 2)
model.load_state_dict(torch.load("models/efficientnet_faces.pth", map_location=device))
model.to(device)
model.eval()

# Label mapping
inv_label_map = {0: "Real", 1: "Fake"}

all_preds = []
all_labels = []

output_lines = ["Filename\tActual\tPredicted\n"]

with torch.no_grad():
    for i, (faces, labels, paths) in enumerate(dataloader):
        B, N, C, H, W = faces.shape
        faces = faces.view(B * N, C, H, W).to(device)
        labels_exp = labels.unsqueeze(1).expand(-1, N).reshape(-1).to(device)
        outputs = model(faces)
        _, predicted = torch.max(outputs, 1)

        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels_exp.cpu().numpy())

        # Logging for each sample
        for j in range(B):
            pred_label = torch.mode(predicted[j*N:(j+1)*N]).values.item()
            actual_label = labels[j].item()
            filename = os.path.basename(paths[j])
            actual_str = inv_label_map[actual_label]
            predicted_str = inv_label_map[pred_label]
            output_lines.append(f"{filename}\t{actual_str}\t{predicted_str}")

# Accuracy, classification report, confusion matrix
acc = accuracy_score(all_labels, all_preds)
cm = confusion_matrix(all_labels, all_preds)
report = classification_report(all_labels, all_preds, target_names=["Real", "Fake"], digits=4)

# Append metrics to output file
output_lines.append(f"\nFinal Accuracy: {acc * 100:.2f}%")
output_lines.append(f"Confusion Matrix:\n{cm}")
output_lines.append(f"\nClassification Report:\n{report}")

# Save predictions and metrics
with open(output_path, "w") as f:
    f.write("\n".join(output_lines))

# Print predictions
print("\n📄 Predictions vs Actual:")
for line in output_lines[1:-3]:
    print(line)

# Print metrics
print(f"\n✅ Test Accuracy: {acc * 100:.2f}%")
print("Confusion Matrix:\n", cm)
print("\n📊 Classification Report:")
print(report)

# Visualize confusion matrix
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Real", "Fake"], yticklabels=["Real", "Fake"])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.tight_layout()

# Save heatmap
cm_plot_path = os.path.join(output_dir, "confusion_matrix.png")
plt.savefig(cm_plot_path)
plt.show()

print(f"\n🖼️ Confusion matrix heatmap saved to: {cm_plot_path}")
print(f"📝 Full evaluation saved to: {output_path}")
