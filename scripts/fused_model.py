import os
import random
import torch
import librosa
import numpy as np
import tensorflow as tf
import subprocess
from facenet_pytorch import MTCNN
from efficientnet_pytorch import EfficientNet
import torchvision.transforms as transforms
from PIL import Image
import cv2


# Paths
TEST_DATA_PATH = "Test_videos"
VIDEO_MODEL_PATH = "models/efficientnet_faces.pth"
AUDIO_MODEL_PATH = "models/audio_deepfake_detector.h5"
FFMPEG_PATH = r"C:\ffmpeg-7.1.1-full\bin\ffmpeg.exe"


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------- Load Models --------------------
def load_video_model():
    model = EfficientNet.from_pretrained('efficientnet-b0')
    model._fc = torch.nn.Linear(model._fc.in_features, 2)
    model.load_state_dict(torch.load(VIDEO_MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

def load_audio_model():
    return tf.keras.models.load_model(AUDIO_MODEL_PATH)

# -------------------- Preprocessing --------------------
mtcnn = MTCNN(image_size=224, margin=0, device=DEVICE)

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

    if len(faces) < num_frames:
        pad_count = num_frames - len(faces)
        pad_tensor = torch.zeros((pad_count, 3, 224, 224), device=DEVICE)
        faces += [pad_tensor[i] for i in range(pad_count)]

    return torch.stack(faces) if faces else None


def extract_audio_from_video(video_path, output_audio_path="temp_audio.wav"):
    try:
        subprocess.run([
            FFMPEG_PATH, "-y",
            "-i", video_path,
            "-vn", "-acodec", "pcm_s16le",
            "-ar", "16000", "-ac", "1",
            output_audio_path
        ], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        print(f"Error extracting audio from {video_path}: {e}")

def extract_mfcc(audio_path, max_pad_length=500):
    y, sr = librosa.load(audio_path, sr=16000)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=100)
    mfcc = (mfcc - np.mean(mfcc)) / np.std(mfcc)
    mfcc = np.pad(mfcc, ((0, 0), (0, max_pad_length - mfcc.shape[1])), mode='constant') if mfcc.shape[1] < max_pad_length else mfcc[:, :max_pad_length]
    return np.expand_dims(mfcc, axis=(0, -1))

# -------------------- Fusion Prediction --------------------
def predict(video_path, video_model, audio_model):
    faces = extract_faces_from_video(video_path)
    if faces is None:
        return None

    faces = faces.to(DEVICE)
    with torch.no_grad():
        outputs = video_model(faces)
        video_scores = torch.softmax(outputs, dim=1)
        video_score = torch.mean(video_scores, dim=0)
        video_real_score = video_score[0].item()
        video_fake_score = video_score[1].item()

    extract_audio_from_video(video_path)
    mfcc = extract_mfcc("temp_audio.wav")
    audio_preds = audio_model.predict(mfcc)
    audio_real_score = audio_preds[0][0]
    audio_fake_score = audio_preds[0][1]

    if os.path.exists("temp_audio.wav"):
        os.remove("temp_audio.wav")

    #fusion_real = 0.5 * video_real_score + 0.5 * audio_real_score
    #fusion_fake = 0.5 * video_fake_score + 0.5 * audio_fake_score
    #predicted_label = "fake" if fusion_real <= fusion_fake else "real"

    # Deepfake means either audio OR video is fake
    if video_fake_score > video_real_score or audio_fake_score > audio_real_score:
        predicted_label = "fake"
    else:
        predicted_label = "real"

    return {
        "predicted_label": predicted_label,
        "audio_real_score": audio_real_score,
        "audio_fake_score": audio_fake_score,
        "video_real_score": video_real_score,
        "video_fake_score": video_fake_score
    }

# -------------------- Run Evaluation --------------------
def select_random_videos(test_data_path, num_videos=10):
    video_files = []
    for label in ["real", "fake"]:
        folder_path = os.path.join(test_data_path, label)
        if os.path.exists(folder_path):
            for f in os.listdir(folder_path):
                if f.endswith(".mp4"):
                    video_files.append((os.path.join(folder_path, f), label))
    return random.sample(video_files, num_videos)

def main():
    video_model = load_video_model()
    audio_model = load_audio_model()
    videos = select_random_videos(TEST_DATA_PATH, num_videos=20)
    correct = 0

    for idx, (video_path, actual_label) in enumerate(videos):
        print(f"\n===== Video {idx+1} =====")
        print(f"Actual Label: {actual_label}")
        result = predict(video_path, video_model, audio_model)
        if result is None:
            print("Skipping video due to face extraction issue.")
            continue
        print(f"Predicted Label: {result['predicted_label']}")
        print(f"Audio - Real: {result['audio_real_score']:.3f}, Fake: {result['audio_fake_score']:.3f}")
        print(f"Video - Real: {result['video_real_score']:.3f}, Fake: {result['video_fake_score']:.3f}")
        if result['predicted_label'] == actual_label:
            correct += 1

    print(f"\n✅ Accuracy: {correct}/{len(videos)} = {(correct / len(videos)) * 100:.2f}%")

if __name__ == "__main__":
    main()
