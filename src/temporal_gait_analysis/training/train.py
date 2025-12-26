# src/temporal_gait_analysis/training/train.py

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence, Optional

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from temporal_gait_analysis.datasets.casia_b import CasiaBDataset
from temporal_gait_analysis.datasets.sampler import TripletSampler
from temporal_gait_analysis.models.temporal_model import TemporalGaitModel


# helpers

def batch_hard_triplet_loss_monitored(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    margin: float = 0.2
):
    # embeddings: [B, D] (assumed normalized)
    dot_product = torch.matmul(embeddings, embeddings.t())
    square_norm = torch.diag(dot_product)
    distances = square_norm.unsqueeze(0) - 2.0 * dot_product + square_norm.unsqueeze(1)
    distances = torch.clamp(distances, min=1e-6).sqrt()

    labels_equal = labels.unsqueeze(0) == labels.unsqueeze(1)
    mask_anchor_positive = labels_equal.float()
    # mask_anchor_negative = 1 - mask_anchor_positive  # unused directly

    pos_dists = (distances * mask_anchor_positive).max(dim=1)[0]
    neg_dists = (distances + mask_anchor_positive * 1e6).min(dim=1)[0]

    triplet_loss = torch.clamp(pos_dists - neg_dists + margin, min=0.0)
    return triplet_loss.mean(), pos_dists.mean(), neg_dists.mean()


def _device_from_arg(device: str) -> torch.device:
    if device == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cpu":
        return torch.device("cpu")
    # allow "cuda:0" etc.
    return torch.device(device)


def _set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# config carrier 

@dataclass
class TrainConfig:
    # data
    data_root: Path
    split_json: Path
    seq_len: int = 30
    augmentation: bool = True

    # model
    part_mode: int = 16          # 1,4,8,16
    encoding: str = "rpe"        # "absolute","cpe","cycle","rpe","sinusoidal"
    num_classes: int = 74

    # batching
    batch_p: int = 8
    batch_k: int = 4
    num_workers: int = 4

    # optimization
    num_epochs: int = 60
    lr_start: float = 1e-3
    lr_milestones: Sequence[int] = (30, 50)
    lr_gamma: float = 0.1
    weight_decay: float = 1e-4
    grad_clip: float = 2.0

    # losses
    label_smoothing: float = 0.1
    triplet_margin: float = 0.2
    triplet_warmup_epochs: int = 5

    # misc
    outdir: Path = Path("runs")
    exp_name: Optional[str] = None
    save_every: int = 5
    device: str = "cuda"
    seed: int = 42
    amp: bool = True


def run_train(args) -> None:
    """
    expects args fields (from cli):
      - data_root
      - split_json
      - outdir
      - device
      - seed
    optional:
      - part_mode
      - encoding
      - batch_p
      - batch_k
      - num_epochs
      - lr_start
      - amp
    """
    # build config from args with sensible defaults 
    cfg = TrainConfig(
        data_root=Path(args.data_root),
        split_json=Path(args.split_json),
        outdir=Path(getattr(args, "outdir", "runs")),
        device=getattr(args, "device", "cuda"),
        seed=int(getattr(args, "seed", 42)),
        part_mode=int(getattr(args, "part_mode", 16)),
        encoding=str(getattr(args, "encoding", "rpe")),
        batch_p=int(getattr(args, "batch_p", 8)),
        batch_k=int(getattr(args, "batch_k", 4)),
        num_epochs=int(getattr(args, "num_epochs", 60)),
        lr_start=float(getattr(args, "lr_start", 1e-3)),
        amp=bool(getattr(args, "amp", True)),
    )

    # name experiment directory deterministically
    exp_name = cfg.exp_name or f"checkpoints_parts{cfg.part_mode}_{cfg.encoding}"
    ckpt_dir = cfg.outdir / exp_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    device = _device_from_arg(cfg.device)
    _set_seed(cfg.seed)

    batch_size = cfg.batch_p * cfg.batch_k

    print(f"Device: {device} | Config: Parts={cfg.part_mode}, Enc={cfg.encoding}")
    print(f"LR Schedule: Start={cfg.lr_start}, Drops at {list(cfg.lr_milestones)}")

    # data 
    print("Loading dataset (augmentation=True)...")
    train_ds = CasiaBDataset(
        cfg.data_root,
        cfg.split_json,
        split="train",
        seq_len=cfg.seq_len,
        augmentation=cfg.augmentation,
    )
    sampler = TripletSampler(train_ds, p_interval=cfg.batch_p, k_samples=cfg.batch_k)
    train_loader = DataLoader(
        train_ds,
        sampler=sampler,
        batch_size=batch_size,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    # model
    model = TemporalGaitModel(
        num_classes=cfg.num_classes,
        part_config=cfg.part_mode,
        encoding=cfg.encoding
    ).to(device)

    real_parts = int(model.num_parts)

    # optim 
    optimizer = optim.AdamW(model.parameters(), lr=cfg.lr_start, weight_decay=cfg.weight_decay)
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=list(cfg.lr_milestones), gamma=cfg.lr_gamma
    )

    # AMP
    scaler = torch.amp.GradScaler("cuda") if (cfg.amp and device.type == "cuda") else None

    ce_loss_fn = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)

    print("Training...")
    model.train()

    for epoch in range(1, cfg.num_epochs + 1):
        acc_ce = 0.0
        acc_trip = 0.0
        acc_pos = 0.0
        acc_neg = 0.0
        correct = 0
        total_samples = 0

        use_triplet = (epoch > cfg.triplet_warmup_epochs)
        current_lr = optimizer.param_groups[0]["lr"]

        pbar = tqdm(train_loader, desc=f"Ep {epoch}/{cfg.num_epochs} [LR={current_lr:.1e}]")

        for i, (inputs, labels, _) in enumerate(pbar):
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            # forward + CE (AMP allowed)
            if scaler is not None:
                with torch.amp.autocast("cuda"):
                    part_embeddings, part_logits = model(inputs)
                    loss_ce_tot = 0.0
                    for k in range(len(part_logits)):
                        loss_ce_tot += ce_loss_fn(part_logits[k], labels)
            else:
                part_embeddings, part_logits = model(inputs)
                loss_ce_tot = 0.0
                for k in range(len(part_logits)):
                    loss_ce_tot += ce_loss_fn(part_logits[k], labels)

            # triplet in fp32 for stability (your original intent)
            loss_trip_tot = 0.0
            batch_pos = 0.0
            batch_neg = 0.0
            if use_triplet:
                for k in range(len(part_logits)):
                    emb_32 = part_embeddings[k].float()
                    feat_norm = F.normalize(emb_32, p=2, dim=1)
                    l_trip, p_d, n_d = batch_hard_triplet_loss_monitored(
                        feat_norm, labels, margin=cfg.triplet_margin
                    )
                    loss_trip_tot += l_trip
                    batch_pos += float(p_d.item())
                    batch_neg += float(n_d.item())

            loss = (loss_ce_tot + loss_trip_tot) / real_parts

            # backward + step
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.grad_clip)
                optimizer.step()

            # metrics
            acc_ce += float(loss_ce_tot.item()) / real_parts
            if use_triplet:
                acc_trip += float(loss_trip_tot.item()) / real_parts
                acc_pos += batch_pos / real_parts
                acc_neg += batch_neg / real_parts

            final_logits = torch.stack(part_logits).sum(dim=0)
            preds = final_logits.argmax(dim=1)
            correct += int((preds == labels).sum().item())
            total_samples += int(labels.size(0))

            pbar.set_postfix_str(f"CE:{acc_ce/(i+1):.2f}|Acc:{correct/total_samples:.1%}")

        scheduler.step()

        # epoch report
        n = len(train_loader)
        print(f"\n--- EPOCH {epoch} (parts={cfg.part_mode}, enc={cfg.encoding}) ---")
        print(f"LR:            {current_lr:.1e}")
        print(f"Accuracy:      {correct/total_samples:.2%}")
        print(f"Loss CE mean:  {acc_ce/n:.4f}")

        if use_triplet:
            avg_p = acc_pos / n
            avg_n = acc_neg / n
            print(f"Loss Triplet:  {acc_trip/n:.4f}")
            print(f"Pos dist:      {avg_p:.4f} (target -> 0)")
            print(f"Neg dist:      {avg_n:.4f} (target -> >{avg_p+cfg.triplet_margin:.4f})")
            print(f"Real margin:   {avg_n - avg_p:.4f}")
        else:
            print("Loss Triplet:  WARMUP")
        print("-" * 40)

        # checkpoint
        if (epoch % cfg.save_every == 0) or (epoch == cfg.num_epochs):
            save_path = ckpt_dir / f"epoch_{epoch}.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "config": {
                        "part_mode": cfg.part_mode,
                        "encoding": cfg.encoding,
                        "seq_len": cfg.seq_len,
                        "batch_p": cfg.batch_p,
                        "batch_k": cfg.batch_k,
                    },
                },
                save_path,
            )
            print(f"Checkpoint: {save_path}")
