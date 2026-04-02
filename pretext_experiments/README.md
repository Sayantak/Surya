# ☀️ Surya Pretext Experiments: Time Advancement vs Band-to-Band

This repository contains an experimental framework built on top of the Surya foundation model for heliophysics. It focuses on evaluating alternative self-supervised pretraining objectives and their impact on downstream tasks.

The primary goal is to compare:

- Time Advancement (original Surya objective)
- Band-to-Band Translation (cross-channel reconstruction)

and study how these affect Active Region Segmentation.

---

## 📖 Overview

Surya is a 366M-parameter spatiotemporal transformer trained on multi-channel Solar Dynamics Observatory (SDO) data. The original model uses time advancement, where future solar states are predicted from past observations.

This project extends that framework by introducing band-to-band translation, where missing spectral channels are reconstructed from available ones at the same timestep.

Key findings from the experiments:

- Band-to-band provides a more stable learning signal during pretraining
- Time advancement can degrade under limited data
- Downstream performance remains similar due to strong finetuning capacity and small data scale

---

## 🧠 Pretraining Objectives

### Time Advancement
- Predict future solar frame from past frames
- Objective: Mean Squared Error (MSE)
- Learns temporal dynamics

### Band-to-Band Translation
- Mask subset of channels and reconstruct them
- Objective: Masked MSE
- Learns cross-channel relationships

Variants:
- AIA-only (harder, intra-modality)
- All channels (easier, cross-modal)

---

## 🚀 Setup

### 1. Clone repository

```bash
git clone https://github.com/Sayantak/Surya/tree/main
cd pretext_experiments
```

### 2. Install dependencies (recommended: uv)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh \
source ~/.bashrc
```

```bash
uv sync \
source .venv/bin/activate
```

---

## 📊 Data

- Pretraining: SDO (13 channels: AIA + HMI)
- Downstream: Active Region segmentation masks

Typical setup:
- Pretraining date: 20241231
- Downstream date: 20241129

Splits:
- 80% train / 10% val / 10% test (time-based)

---

## 🧪 Running Experiments

### Time Advancement (Pretraining)

```bash
uv run python -u pretext_experiments/scripts/run_time_advancement.py
  --prepare-data
  --download-date 20241231
  --n-steps 500
```

---

### Band-to-Band (Pretraining)

```bash
uv run python -u pretext_experiments/scripts/run_band_to_band.py
  --prepare-data
  --download-date 20241231
  --min-masked-channels 1
  --max-masked-channels 3
  --n-steps 500
```

AIA-only variant:

--exclude-hmi-from-targets

---

### Evaluate Original Surya Checkpoint

```bash
uv run python -u pretext_experiments/scripts/run_time_advancement.py
  --prepare-data
  --download-date 20241231
  --eval-only
  --model-dir pretext_experiments/data/Surya-1.0
```

---

### Active Region Segmentation (Downstream)

```bash
uv run python -u pretext_experiments/scripts/run_ar_seg.py \
  --prepare-sdo-data \
  --prepare-ar-data \
  --download-date 20241129 \
  --restrict-date 20241129 \
  --init-ckpt <checkpoint.pt> \
  --finetune-mode full \
  --n-steps 500
```

---

## 📈 Evaluation

Metrics:

Pretraining:
- MSE (masked for band-to-band)

Downstream:
- Dice
- IoU

Observations:

- Band-to-band improves training stability
- No consistent downstream improvement
- Full finetuning overrides pretraining differences
- LoRA shows slight sensitivity to pretraining

---

## 🧭 Future Work

- Scale pretraining to multi-month datasets
- Use linear probing for cleaner evaluation
- Explore task-aligned pretraining objectives
- Combine temporal + cross-channel objectives

---
