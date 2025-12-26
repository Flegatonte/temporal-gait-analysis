# src/temporal_gait_analysis/evaluation/test.py

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import math
import random

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from temporal_gait_analysis.datasets.casia_b import CasiaBDataset
from temporal_gait_analysis.models.temporal_model import TemporalGaitModel


# optional augmentation (kept, but OFF by default)

def _random_erasing_2d(img2d: torch.Tensor, sl: float = 0.02, sh: float = 0.2, r1: float = 0.3) -> torch.Tensor:
    # img2d: [H, W]
    H, W = img2d.shape
    area = H * W

    target_area = random.uniform(sl, sh) * area
    aspect_ratio = random.uniform(r1, 1.0 / r1)

    eh = int(round(math.sqrt(target_area * aspect_ratio)))
    ew = int(round(math.sqrt(target_area / aspect_ratio)))

    if 0 < eh < H and 0 < ew < W:
        x1 = random.randint(0, H - eh)
        y1 = random.randint(0, W - ew)
        img2d[x1:x1 + eh, y1:y1 + ew] = 0

    return img2d


def augment_probe_like_training(x: torch.Tensor) -> torch.Tensor:
    """
    x: Tensor [1, T, H, W] (C=1).
    Apply horizontal flip and random erasing with probability 0.5
    """
    if random.random() < 0.5:
        x = torch.flip(x, dims=[-1])

    if random.random() < 0.5:
        C, T, H, W = x.shape
        for t in range(T):
            frame = x[0, t]
            x[0, t] = _random_erasing_2d(frame)

    return x

# config carrier

@dataclass
class TestConfig:
    data_root: Path
    split_json: Path
    ckpt: Path

    eval_protocol: str = "center_crop"   # "center_crop" or "full_seq"
    device: str = "cuda"

    # overrides; if None, read from checkpoint config
    part_mode_override: Optional[int] = None
    encoding_override: Optional[str] = None

    # dataset settings
    seq_len: int = 30
    num_workers: int = 2

    # debug/experimental
    apply_aug_to_probe_in_eval: bool = False


def _device_from_arg(device: str) -> torch.device:
    if device == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cpu":
        return torch.device("cpu")
    return torch.device(device)


def _load_checkpoint(ckpt_path: Path, device: torch.device) -> dict:
    if not ckpt_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")
    return torch.load(ckpt_path, map_location=device)


def extract_features(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    protocol: str,
    apply_aug_to_probe_in_eval: bool = False
) -> Tuple[torch.Tensor, np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    features = []
    labels = []
    cam_views = []
    seq_types = []

    print(f"Extracting features (protocol: {protocol})...")

    gallery_types = {"nm-01", "nm-02", "nm-03", "nm-04"}

    with torch.no_grad():
        for inputs, lbls, meta in tqdm(loader):
            inputs = inputs.to(device, non_blocking=True)

            # optional: apply augmentation to probes only (OFF by default)
            if apply_aug_to_probe_in_eval:
                if inputs.ndim == 4:
                    inputs = inputs.unsqueeze(1)  # [B,T,H,W] -> [B,1,T,H,W]

                for b, seq_type in enumerate(meta["type"]):
                    is_probe = (seq_type not in gallery_types)
                    if is_probe:
                        inputs[b] = augment_probe_like_training(inputs[b])

            # feature extraction
            if protocol == "full_seq":
                # expects dataset returns a single sequence per item -> batch_size=1
                # inputs: [1, T_tot, H, W] or [1, 1, T_tot, H, W]
                if inputs.ndim == 5:
                    inputs = inputs.squeeze(0)  # [1,1,T,H,W] -> [1,T,H,W] or [1,T,H,W] if C=1
                else:
                    inputs = inputs.squeeze(0)  # [1,T,H,W] -> [T,H,W]??? depends, handle below

                # normalize to [T, 1, H, W]
                if inputs.ndim == 3:
                    # [T,H,W] -> [T,1,H,W]
                    inputs = inputs.unsqueeze(1)
                elif inputs.ndim == 4:
                    # could be [1,T,H,W] or [T,1,H,W]
                    if inputs.shape[0] == 1:
                        inputs = inputs.permute(1, 0, 2, 3)  # [1,T,H,W] -> [T,1,H,W]

                T_total = inputs.shape[0]
                seq_len = 30
                stride = seq_len
                chunk_embeddings = []

                for t in range(0, T_total, stride):
                    clip = inputs[t:t + seq_len]
                    if clip.shape[0] < seq_len:
                        if T_total >= seq_len:
                            clip = inputs[-seq_len:]
                        else:
                            continue

                    # [T,1,H,W] -> [1,1,T,H,W]
                    clip_batch = clip.permute(1, 0, 2, 3).unsqueeze(0)
                    emb = model(clip_batch)  # expected [1, D]
                    chunk_embeddings.append(emb)

                if len(chunk_embeddings) > 0:
                    final_feat = torch.mean(torch.stack(chunk_embeddings), dim=0)
                    final_feat = torch.nn.functional.normalize(final_feat, p=2, dim=1)
                    features.append(final_feat.cpu())
                else:
                    # fallback, should rarely happen
                    feat_dim = getattr(model, "feat_dim", 256)
                    num_parts = getattr(model, "num_parts", 1)
                    features.append(torch.zeros(1, feat_dim * num_parts))
            else:
                # center_crop: expects inputs already as fixed clip
                # ensure [B,1,T,H,W]
                if inputs.ndim == 4:
                    inputs = inputs.unsqueeze(1)

                emb = model(inputs)  # expected [B, D]
                features.append(emb.cpu())

            # metadata
            labels.extend(lbls.numpy())
            cam_views.extend(meta["view"])
            seq_types.extend(meta["type"])

    if len(features) == 0:
        raise RuntimeError("no features extracted")

    features = torch.cat(features, dim=0)

    # alignment check
    if len(features) != len(seq_types):
        min_len = min(len(features), len(seq_types))
        features = features[:min_len]
        labels = labels[:min_len]
        cam_views = cam_views[:min_len]
        seq_types = seq_types[:min_len]

    return features, np.array(labels), np.array(cam_views), np.array(seq_types)


def compute_rank_metrics(features: torch.Tensor, labels: np.ndarray, views: np.ndarray, types: np.ndarray, protocol: str) -> None:
    is_gallery = np.array([t in ["nm-01", "nm-02", "nm-03", "nm-04"] for t in types], dtype=bool)
    is_probe = ~is_gallery

    if not is_gallery.any():
        raise RuntimeError("no gallery samples found")

    gal_feats = features[is_gallery]
    gal_labels = labels[is_gallery]
    gal_views = views[is_gallery]

    probe_feats = features[is_probe]
    probe_labels = labels[is_probe]
    probe_views = views[is_probe]
    probe_types = types[is_probe]

    print(f"\n--- RANK RESULTS ({protocol}) ---")
    print(f"Gallery: {len(gal_feats)} | Probe: {len(probe_feats)}")

    # similarity matrix probe x gallery
    similarity = torch.matmul(probe_feats, gal_feats.t())

    conditions = {
        "NM": ["nm-05", "nm-06"],
        "BG": ["bg-01", "bg-02"],
        "CL": ["cl-01", "cl-02"],
    }

    print(f"{'Cond':<4} | {'Rank-1':<8} | {'Rank-5':<8}")
    print("-" * 25)

    for cond_name, cond_seqs in conditions.items():
        cond_mask = np.isin(probe_types, cond_seqs)
        if not cond_mask.any():
            continue

        p_sim = similarity[cond_mask]
        p_lbl = probe_labels[cond_mask]
        p_view = probe_views[cond_mask]

        correct_r1 = 0
        correct_r5 = 0
        total = 0

        for i in range(len(p_lbl)):
            sims = p_sim[i]
            true_label = p_lbl[i]

            # exclude same-view matches
            valid_mask = (gal_views != p_view[i])
            if not valid_mask.any():
                continue

            valid_sims = sims[valid_mask]
            valid_gal_labels = gal_labels[valid_mask]

            k = min(5, len(valid_sims))
            _, top_indices = torch.topk(valid_sims, k=k)
            top_labels = valid_gal_labels[top_indices]

            if true_label == top_labels[0]:
                correct_r1 += 1
            if true_label in top_labels:
                correct_r5 += 1
            total += 1

        acc_r1 = (correct_r1 / total) * 100 if total > 0 else 0.0
        acc_r5 = (correct_r5 / total) * 100 if total > 0 else 0.0
        print(f"{cond_name:<4} | {acc_r1:<6.2f}% | {acc_r5:<6.2f}%")


def run_test(args) -> None:
    """
    expects args fields (from cli):
      - data_root
      - split_json
      - ckpt
    optional:
      - device
      - eval_protocol  (center_crop|full_seq)
      - part_mode_override
      - encoding_override
      - apply_aug_to_probe_in_eval
    """
    cfg = TestConfig(
        data_root=Path(args.data_root),
        split_json=Path(args.split_json),
        ckpt=Path(args.ckpt),
        device=getattr(args, "device", "cuda"),
        eval_protocol=getattr(args, "eval_protocol", "center_crop"),
        part_mode_override=getattr(args, "part_mode_override", None),
        encoding_override=getattr(args, "encoding_override", None),
        apply_aug_to_probe_in_eval=bool(getattr(args, "apply_aug_to_probe_in_eval", False)),
    )

    device = _device_from_arg(cfg.device)

    checkpoint = _load_checkpoint(cfg.ckpt, device)
    saved_config = checkpoint.get("config", {})

    part_mode = cfg.part_mode_override if cfg.part_mode_override is not None else saved_config.get("part_mode", 4)
    encoding = cfg.encoding_override if cfg.encoding_override is not None else saved_config.get("encoding", "absolute")

    print(f"Loading checkpoint: {cfg.ckpt}")
    print(f"Detected config -> Parts: {part_mode}, Encoding: {encoding}")
    print(f"Device: {device}")

    model = TemporalGaitModel(num_classes=74, part_config=part_mode, encoding=encoding).to(device)

    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    # dataset / loader
    eval_mode = cfg.eval_protocol
    batch_size = 1 if eval_mode == "full_seq" else 64

    print(f"Testing protocol: {eval_mode} (batch size: {batch_size})")

    test_ds = CasiaBDataset(
        cfg.data_root,
        cfg.split_json,
        split="test",
        seq_len=cfg.seq_len,
        eval_mode=eval_mode,
    )

    test_loader = DataLoader(test_ds, batch_size=batch_size, num_workers=cfg.num_workers, shuffle=False)

    feats, labels, views, types = extract_features(
        model,
        test_loader,
        device,
        protocol=eval_mode,
        apply_aug_to_probe_in_eval=cfg.apply_aug_to_probe_in_eval,
    )

    compute_rank_metrics(feats, labels, views, types, protocol=eval_mode)
