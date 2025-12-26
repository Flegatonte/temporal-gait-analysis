# src/temporal_gait_analysis/tools/make_splits.py

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List


@dataclass
class SplitConfig:
    data_root: Path
    out_json: Path


def _is_image_dir(view_dir: Path) -> bool:
    if not view_dir.is_dir():
        return False
    imgs = list(view_dir.glob("*.png")) + list(view_dir.glob("*.jpg")) + list(view_dir.glob("*.jpeg"))
    return len(imgs) > 0


def run_make_splits(args) -> None:
    """
    Build CASIA-B LT split (Large-sample Training):
      - train: subjects 001-074
      - test : subjects 075-124

    expects args fields:
      - data_root (dataset root, pointing to .../CASIA-B/output)
      - out (output json path)
    """
    cfg = SplitConfig(
        data_root=Path(args.data_root),
        out_json=Path(args.out),
    )

    if not cfg.data_root.exists():
        raise FileNotFoundError(f"data_root does not exist: {cfg.data_root}")

    train_ids = {f"{i:03d}" for i in range(1, 75)}
    test_ids = {f"{i:03d}" for i in range(75, 125)}

    splits: Dict[str, List[str]] = {"train": [], "test": []}

    print(f"Scanning: {cfg.data_root}")

    for subject_dir in sorted(cfg.data_root.iterdir()):
        if not subject_dir.is_dir():
            continue

        sid = subject_dir.name  # "001", ...

        if sid in train_ids:
            split_key = "train"
        elif sid in test_ids:
            split_key = "test"
        else:
            # ignore anything outside CASIA-B 001-124
            continue

        # subject/condition/view
        for cond_dir in sorted(subject_dir.iterdir()):  # nm-01, bg-01...
            if not cond_dir.is_dir():
                continue

            for view_dir in sorted(cond_dir.iterdir()):  # 000, 018...
                if not _is_image_dir(view_dir):
                    continue

                # relative path from data_root
                rel_path = f"{sid}/{cond_dir.name}/{view_dir.name}"
                splits[split_key].append(rel_path)

    print(f"TRAIN sequences: {len(splits['train'])} (subjects 001-074)")
    print(f"TEST  sequences: {len(splits['test'])} (subjects 075-124)")

    cfg.out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(cfg.out_json, "w", encoding="utf-8") as f:
        json.dump(splits, f, indent=4)

    print(f"Saved split file: {cfg.out_json}")
