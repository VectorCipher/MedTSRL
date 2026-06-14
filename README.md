# MedTSRL: Medical Tutor-Student Reinforcement Learning Framework

## Overview

MedTSRL is a novel Tutor-Student Reinforcement Learning framework designed to improve both **classification** and **localization** performance in medical deepfake detection tasks.

The framework leverages a **Teacher Model (U-Net)** to provide manipulation localization guidance and a **Student Model (TransUNet)** that learns through reinforcement learning-based optimization. The goal is not only to classify whether a medical image is real or manipulated but also to accurately localize the manipulated regions.

---

# Problem Statement

Most existing medical deepfake detection systems focus primarily on image-level classification and often fail to precisely identify manipulated regions.

Challenges include:

* High classification accuracy but poor localization.
* Lack of explainability for clinical usage.
* Difficulty detecting subtle manipulation artifacts.
* Weak spatial understanding of forged regions.

---

# Framework Architecture

## Teacher Model

**U-Net**

Responsibilities:

* Generate manipulation localization masks.
* Provide spatial guidance to the student model.
* Act as an expert policy during training.

The teacher model learns pixel-level manipulation patterns and produces localization maps used as supervision signals.

---

## Student Model

**TransUNet**

Responsibilities:

* Learn manipulation classification.
* Learn localization simultaneously.
* Improve upon teacher predictions using reinforcement learning.

The TransUNet combines:

* CNN-based local feature extraction
* Vision Transformer global context modeling
* Segmentation decoder for localization

This architecture allows the model to capture both fine-grained texture artifacts and long-range dependencies.

---

# Reinforcement Learning Strategy

The student model is optimized using a reinforcement learning paradigm where rewards are assigned based on:

### Classification Reward

Encourages correct identification of:

* Real Images
* Manipulated Images

Metrics considered:

* Accuracy
* Precision
* Recall
* F1 Score

---

### Localization Reward

Encourages accurate localization of manipulated regions.

Metrics considered:

* IoU (Intersection over Union)
* Localization Precision
* Localization Recall
* Localization F1 Score

---

### Teacher Guidance

The teacher-generated localization masks act as expert demonstrations, enabling the student to learn more robust manipulation representations.

---

# Experimental Results

## Teacher Model (U-Net)

### Classification Performance

| Metric    | Score  |
| --------- | ------ |
| Accuracy  | 0.9382 |
| Precision | 1.0000 |
| Recall    | 0.9207 |
| F1 Score  | 0.9587 |

### Confusion Matrix

| Actual / Predicted | Real (0) | Fake (1) |
| ------------------ | -------- | -------- |
| Real (0)           | 118      | 0        |
| Fake (1)           | 33       | 383      |

### Localization Performance

| Metric                 | Score  |
| ---------------------- | ------ |
| Localization Precision | 0.0199 |
| Localization Recall    | 0.0125 |
| Localization F1@0.5    | 0.0154 |
| Mean IoU               | 0.5369 |
| Count MAE              | 0.7266 |

---

## Student Model (TransUNet + Tutor-Student RL)


### Classification Performance

| Metric    | Score  |
| --------- | ------ |
| Accuracy  | 0.9963 |
| Precision | 1.0000 |
| Recall    | 0.9952 |
| F1 Score  | 0.9976 |

### Confusion Matrix

| Actual / Predicted | Real (0) | Fake (1) |
| ------------------ | -------- | -------- |
| Real (0)           | 118      | 0        |
| Fake (1)           | 2        | 414      |

### Localization Performance

| Metric                 | Score  |
| ---------------------- | ------ |
| Localization Precision | 0.8722 |
| Localization Recall    | 0.7905 |
| Localization F1@0.5    | 0.8294 |
| Mean IoU               | 0.6870 |
| Count MAE              | 0.2453 |

---

# Performance Improvement

## Classification Improvement

| Metric   | U-Net  | MedTSRL (TransUNet) |
| -------- | ------ | ------------------- |
| Accuracy | 93.82% | 99.63%              |
| Recall   | 92.07% | 99.52%              |
| F1 Score | 95.87% | 99.76%              |

---

## Localization Improvement

| Metric                 | U-Net  | MedTSRL (TransUNet) |
| ---------------------- | ------ | ------------------- |
| Localization Precision | 0.0199 | 0.8722              |
| Localization Recall    | 0.0125 | 0.7905              |
| Localization F1@0.5    | 0.0154 | 0.8294              |
| Mean IoU               | 0.5369 | 0.6870              |
| Count MAE              | 0.7266 | 0.2453              |

---

# Key Findings

1. The Tutor-Student Reinforcement Learning strategy significantly improves localization performance.

2. The student model successfully learns richer manipulation representations through teacher-guided reinforcement learning.

3. Classification accuracy reaches near-perfect performance while maintaining strong localization capabilities.

4. Localization F1 improves from **1.54% to 82.94%**, demonstrating substantial gains in manipulation region detection.

5. The framework provides improved explainability by highlighting manipulated regions rather than producing only image-level predictions.

---

# Applications

* Medical Deepfake Detection
* Radiology Image Verification
* Clinical Decision Support
* Healthcare AI Security
* Medical Image Forensics
* Explainable Medical AI

---

# How to Run

To replicate these results or train on your own medical deepfake dataset, follow these steps:

### 1. Dataset Structure
Dataset url : [Medical Image Deepfake Detection Dataset](https://drive.google.com/drive/folders/1B01A9rLshs1_-I3v5zuM-nfjEkU5G3Q8?usp=sharing)
Ensure your dataset directory is structured as follows:
```text
Dataset_deepfake_detection/
├── images/     (Original/Manipulated 128x128 images)
├── density/    (Density maps from Teacher U-Net)
├── masks/      (Binary forgery masks)
└── labels/     (Bounding box coordinates)
```

### 2. Training the L-TSRL Framework
Run the main training script. The training automatically handles the 3-stage curriculum (Warmup -> Behavioral Cloning -> PPO Reinforcement Learning).

```bash
python train_ltsrl.py \
  --data_dir "/path/to/Dataset_deepfake_detection" \
  --resume "/path/to/pretrained/TransUnet.pth" \
  --epochs 60 \
  --warmup_epochs 10 \
  --bc_epochs 10 \
  --kl_coef 0.05 \
  --batch_size 16
```

#### Training Stages Explained:
1. **Warmup (Epochs 1-10):** The TransUNet student trains normally using the Teacher's density maps to establish a baseline.
2. **Behavioral Cloning (Epochs 11-20):** The Tutor PPO agent is pre-trained to mimic an expert heuristic difficulty score using Mean Squared Error (MSE), resolving the RL cold-start problem.
3. **Joint RL (Epochs 21-60):** The Tutor actively scales the localization loss per-sample using PPO to maximize the overall Localization F1 score, anchored by a KL divergence constraint (`--kl_coef`).

---

