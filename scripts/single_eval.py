import os
import torch
import cv2
import numpy as np
from PIL import Image
from facenet_pytorch import MTCNN
from efficientnet_pytorch import EfficientNet

# ==== Configuration ====
video_path = "test.mp4"  # Single video path
model_path = "models/efficientnet_faces.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==== Initialize MTCNN ====
mtcnn = MTCNN(image_size=224, margin=0, device=device)

# ==== Load Trained Model ====
model = EfficientNet.from_pretrained('efficientnet-b0')
model._fc = torch.nn.Linear(model._fc.in_features, 2)
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# ==== Face Extraction ====
def extract_faces_from_video(video_path, num_frames=5):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    faces = []

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        face = mtcnn(img)
        if face is not None:
            faces.append(face)
    cap.release()

    if faces:
        if len(faces) < num_frames:
            pad_count = num_frames - len(faces)
            pad_tensor = torch.zeros((pad_count, 3, 224, 224), device=device)
            faces += [pad_tensor[i] for i in range(pad_count)]
        return torch.stack(faces)
    return None

# ==== Prediction ====
print(f"\n🎬 Processing: {video_path}")
faces = extract_faces_from_video(video_path)

if faces is None:
    print("❌ No faces extracted from the video.")
else:
    faces = faces.to(device)
    with torch.no_grad():
        outputs = model(faces)
        probs = torch.softmax(outputs, dim=1)
        avg_probs = torch.mean(probs, dim=0)
        predicted_class = torch.argmax(avg_probs).item()

    inv_label_map = {0: "Real", 1: "Fake"}
    print(f"\n✅ Prediction: {inv_label_map[predicted_class].upper()}")
    print(f"🔢 Confidence Scores:")
    print(f"    🎭 Fake: {avg_probs[1].item():.4f}")
    print(f"    ✅ Real: {avg_probs[0].item():.4f}")
