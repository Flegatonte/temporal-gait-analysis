# Temporal Gait Analysis

This repository contains the code for **TemporalGaitModel**, a silhouette-based gait recognition model designed to explicitly model temporal dynamics in human gait sequences.  
The project follows the standard **CASIA-B evaluation protocol** and is intended for reproducible academic experimentation.

---

## Dataset

We use the **CASIA-B gait dataset**, a standard benchmark for gait recognition provided by the Chinese Academy of Sciences.

CASIA-B contains silhouette image sequences of **124 subjects**, recorded under:
- three walking conditions: normal walking (NM), walking with a bag (BG), and walking with a coat (CL);
- eleven camera viewpoints spanning from 0° to 180°.

In this project, **raw videos are not used**.  
Instead, we assume that silhouette frames have already been extracted and organised as temporal image sequences.

You can download the preprocessed dataset here: https://www.kaggle.com/datasets/trnquanghuyn/casia-b

### Expected directory structure

<DATA_ROOT>/
001/nm-01/000/.png
001/nm-01/018/.png
...
124/cl-02/180/*.png


Each folder `<subject>/<sequence>/<view>/` corresponds to one gait sequence.

---

## Train / Test split

We follow the **Large Training (LT) protocol** of CASIA-B:

- **Training subjects:** 001–074  
- **Test subjects:** 075–124  

No validation set is introduced.  
All hyperparameters are fixed in advance, and test identities are never seen during training.

A split file can be generated automatically using the provided script.

---

## Installation

Create and activate a Python environment (Python ≥ 3.9 recommended):

```bash
conda create -n temporal-gait python=3.10 -y
conda activate temporal-gait

Install the project in editable mode:

pip install -e .

PyTorch must be installed separately according to your system configuration.

Usage
Create CASIA-B LT splits

python -m temporal_gait_analysis.cli make-splits \
  --data-root <DATA_ROOT> \
  --out data/casia_b_splits_lt.json

Training (example run)

python -m temporal_gait_analysis.cli train \
  --data-root <DATA_ROOT> \
  --split-json data/casia_b_splits_lt.json \
  --outdir runs \
  --device cuda

Evaluation

Evaluation follows the cross-view identification protocol:
normal-walking sequences (NM) form the gallery, while all other conditions (BG, CL) are used as probes, excluding same-view matches.

python -m temporal_gait_analysis.cli test \
  --data-root <DATA_ROOT> \
  --split-json data/casia_b_splits_lt.json \
  --ckpt <CHECKPOINT_PATH> \
  --eval-mode center_crop \
  --device cuda

Notes

The model operates on fixed-length temporal clips during training.
At test time, either a central clip or the full sequence can be evaluated.
The code is structured to allow easy modification of temporal encodings and part-based configurations.

License

This project is intended for academic use.