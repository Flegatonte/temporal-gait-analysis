# datasets/gait_dataset.py
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
import json
import random

class CasiaBDataset(Dataset):
    def __init__(self, data_root, split_json, split="train", seq_len=30, resolution=64, augmentation=False, eval_mode="center_crop"):
        """
        Args:
            eval_mode (str): "center_crop" (Default, vecchio metodo) o "full_seq" (SOTA method per test).
                             Questo parametro viene ignorato se split="train".
        """
        self.data_root = Path(data_root)
        self.seq_len = seq_len
        self.resolution = resolution
        self.split = split
        self.augmentation = augmentation
        self.eval_mode = eval_mode # <--- NUOVO PARAMETRO
        
        with open(split_json, "r") as f:
            data = json.load(f)
            self.seq_paths = data[split]
            
        unique_ids = sorted(list(set([p.split("/")[0] for p in self.seq_paths])))
        self.id2label = {sid: i for i, sid in enumerate(unique_ids)}
        
        self.data_list = []
        self.labels = []
        
        for p in self.seq_paths:
            sid = p.split("/")[0]
            label = self.id2label[sid]
            self.data_list.append(p)
            self.labels.append(label)

    def __len__(self):
        return len(self.data_list)

    def augment_sequence(self, seq_imgs):
        # ... (Codice augmentation invariato, lo ometto per brevità ma tu lascialo!) ...
        # Copia pure il metodo augment_sequence dal file precedente
        if random.random() < 0.5:
            seq_imgs = [cv2.flip(img, 1) for img in seq_imgs]
            
        if random.random() < 0.5:
            sl, sh, r1 = 0.02, 0.2, 0.3
            for i in range(len(seq_imgs)):
                img = seq_imgs[i]
                h, w = img.shape
                area = h * w
                target_area = random.uniform(sl, sh) * area
                aspect_ratio = random.uniform(r1, 1/r1)
                eh = int(round(np.sqrt(target_area * aspect_ratio)))
                ew = int(round(np.sqrt(target_area / aspect_ratio)))
                if eh < h and ew < w:
                    x1 = random.randint(0, h - eh)
                    y1 = random.randint(0, w - ew)
                    img[x1:x1+eh, y1:y1+ew] = 0
                    seq_imgs[i] = img
        return seq_imgs

    def load_sequence(self, seq_path):
        full_path = self.data_root / seq_path
        frames_paths = sorted(list(full_path.glob("*.png")) + list(full_path.glob("*.jpg")))
        
        num_frames = len(frames_paths)
        if num_frames == 0:
            # Ritorna tensore vuoto/zero di lunghezza minima per non crashare
            return torch.zeros(1, self.seq_len, self.resolution, self.resolution)
            
        # --- LOGICA DI SAMPLING MODIFICATA ---
        indices = []
        
        if self.split == "train":
            # TRAINING: Sempre Random Sampling (Invariato)
            if num_frames < self.seq_len:
                base_indices = list(range(num_frames))
                while len(indices) < self.seq_len:
                    indices += base_indices
                indices = indices[:self.seq_len]
                indices.sort()
            else:
                start = random.randint(0, num_frames - self.seq_len)
                indices = list(range(start, start + self.seq_len))
                
        else:
            # TESTING
            if self.eval_mode == "full_seq":
                # MODALITÀ SOTA: Prendi TUTTI i frame
                indices = list(range(num_frames))
                # Se il video è enorme (>500 frame), sottocampiona per salvare RAM
                if len(indices) > 500:
                    indices = indices[::2]
            else:
                # MODALITÀ VECCHIA (Center Crop) - Default per compatibilità
                if num_frames < self.seq_len:
                    base_indices = list(range(num_frames))
                    while len(indices) < self.seq_len:
                        indices += base_indices
                    indices = indices[:self.seq_len]
                    indices.sort()
                else:
                    start = (num_frames - self.seq_len) // 2
                    indices = list(range(start, start + self.seq_len))

        # Caricamento immagini
        seq_imgs = []
        for i in indices:
            path = str(frames_paths[i])
            img = cv2.imread(path, 0)
            if img is None:
                img = np.zeros((self.resolution, self.resolution), dtype=np.uint8)
            if img.shape != (self.resolution, self.resolution):
                img = cv2.resize(img, (self.resolution, self.resolution))
            seq_imgs.append(img)
            
        if self.split == "train" and self.augmentation:
            seq_imgs = self.augment_sequence(seq_imgs)
            
        seq_np = np.array(seq_imgs, dtype=np.float32) / 255.0
        seq_tensor = torch.from_numpy(seq_np) 
        seq_tensor = seq_tensor.unsqueeze(0) # [1, T, H, W]
        
        return seq_tensor

    def __getitem__(self, idx):
        seq_path = self.data_list[idx]
        label = self.labels[idx]
        seq = self.load_sequence(seq_path)
        
        meta = {
            "seq_path": seq_path,
            "label_orig": seq_path.split("/")[0],
            "type": seq_path.split("/")[1],
            "view": seq_path.split("/")[2]
        }
        
        return seq, label, meta