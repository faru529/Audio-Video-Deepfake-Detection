import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet
import os
import random

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = EfficientNet.from_pretrained('efficientnet-b0')
model._fc = nn.Linear(model._fc.in_features, 2)
model.load_state_dict(torch.load("models/efficientnet_faces.pth", map_location=device))
model.to(device)
model.eval()

# Label maps
label_map = {
    "RealVideos": 0,
    "FakeVideos": 1
}
inv_label_map = {0: "Real", 1: "Fake"}

# Parameters
num_total_samples = 50
real_ratio = 0.4  # 40% real, 60% fake
num_real = int(num_total_samples * real_ratio)
num_fake = num_total_samples - num_real

# Collect all .pt samples separately
root_dir = "extracted_faces"
real_samples, fake_samples = [], []

for label_name, label in label_map.items():
    folder = os.path.join(root_dir, label_name)
    for root, _, files in os.walk(folder):
        for file in files:
            if file.endswith('.pt'):
                path = os.path.join(root, file)
                if label == 0:
                    real_samples.append((path, label))
                else:
                    fake_samples.append((path, label))

# Randomly sample the desired number
random.shuffle(real_samples)
random.shuffle(fake_samples)
samples_to_predict = real_samples[:num_real] + fake_samples[:num_fake]
random.shuffle(samples_to_predict)  # Mix real and fake

# Output setup
output_dir = "predictions_output"
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "predictions.txt")
output_lines = ["Filename\tActual\tPredicted\n"]

correct = 0

# Predict
with torch.no_grad():
    for path, actual_label in samples_to_predict:
        faces = torch.load(path).to(device)  # [N, 3, 224, 224]
        outputs = model(faces)
        preds = torch.argmax(outputs, dim=1)
        pred_label = torch.mode(preds).values.item()

        actual_str = inv_label_map[actual_label]
        predicted_str = inv_label_map[pred_label]

        if pred_label == actual_label:
            correct += 1

        filename = os.path.basename(path)
        line = f"{filename}\t{actual_str}\t{predicted_str}"
        print(line)
        output_lines.append(line)

# Accuracy
accuracy = 100 * correct / len(samples_to_predict)
output_lines.append(f"\nFinal Accuracy: {accuracy:.2f}% ({correct} out of {len(samples_to_predict)})")
print(f"\nFinal Accuracy: {accuracy:.2f}% ({correct} out of {len(samples_to_predict)})")

# Save to file
with open(output_path, "w") as f:
    f.write("\n".join(output_lines))

print(f"\n✅ Predictions saved to: {output_path}")
