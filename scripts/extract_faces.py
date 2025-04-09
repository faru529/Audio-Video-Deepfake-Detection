import os
import torch
from facenet_pytorch import MTCNN
import cv2
from PIL import Image
import numpy as np

mtcnn = MTCNN(image_size=224, margin=0)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
    return torch.stack(faces) if faces else None

def save_faces(root_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for label_dir in ['RealVideos', 'FakeVideos']:
        full_dir = os.path.join(root_dir, label_dir)
        for root, _, files in os.walk(full_dir):
            for file in files:
                if file.endswith('.mp4'):
                    video_path = os.path.join(root, file)
                    relative_path = os.path.relpath(video_path, root_dir)
                    output_path = os.path.join(output_dir, relative_path.replace('.mp4', '.pt'))
                    os.makedirs(os.path.dirname(output_path), exist_ok=True)
                    if not os.path.exists(output_path):
                        faces = extract_faces_from_video(video_path)
                        if faces is not None:
                            torch.save(faces, output_path)
                            print(f"Saved: {output_path}")

# For training data
#root_dir = "data/FakeAVCeleb"
#output_dir = "extracted_faces"

# For test data
root_dir = "test_videos"
output_dir = "extracted_faces_test"

save_faces(root_dir, output_dir)
