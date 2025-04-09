import os
import torch
from torch.utils.data import Dataset
import random

class FaceTensorDataset(Dataset):
    def __init__(self, root_dir, num_frames=5, oversample=True):
        self.samples = []
        self.labels = []
        self.num_frames = num_frames

        real_samples = []
        fake_samples = []

        # Collect samples
        for label, folder in enumerate(["RealVideos", "FakeVideos"]):
            folder_path = os.path.join(root_dir, folder)
            for root, _, files in os.walk(folder_path):
                for file in files:
                    if file.endswith('.pt'):
                        file_path = os.path.join(root, file)
                        if folder == "RealVideos":
                            real_samples.append((file_path, 0))
                        else:
                            fake_samples.append((file_path, 1))

        # Oversample the minority class
        if oversample:
            max_count = max(len(real_samples), len(fake_samples))
            real_samples = (real_samples * (max_count // len(real_samples) + 1))[:max_count]
            fake_samples = (fake_samples * (max_count // len(fake_samples) + 1))[:max_count]

        all_samples = real_samples + fake_samples
        random.shuffle(all_samples)

        self.samples, self.labels = zip(*all_samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        faces = torch.load(self.samples[idx])  # shape: [N, 3, 224, 224]
        label = self.labels[idx]
        path = self.samples[idx]

        # Pad or truncate to exactly self.num_frames
        if faces.size(0) < self.num_frames:
            pad_count = self.num_frames - faces.size(0)
            pad_tensor = torch.zeros((pad_count, 3, 224, 224))  # black frames
            faces = torch.cat([faces, pad_tensor], dim=0)
        else:
            faces = faces[:self.num_frames]

        return faces, torch.tensor(label, dtype=torch.long), path
