# Project Context: Spatio-Temporal Foundation Model for NIDS
**Role:** Principal ML Engineer & Cybersecurity Researcher
**Objective:** Develop a Self-Supervised Zero-Day Detection system using a Hybrid Masked Transformer.

## 1. Data Specifications
- **Source:** Global Merge of CIC-IDS2018 (Canada) and UNSW-NB15 (Australia).
- **Scale:** 22,480,953 rows × 55 features (NetFlow v3 metadata).
- **Format:** Apache Arrow sharded datasets (via Hugging Face `datasets`).
- **Input Nature:** Tabular network flows treated as Spatio-Temporal sequences.

## 2. Model Architecture
- **Type:** Masked Transformer (Encoder-only, MAE).
- **Task:** Masked Feature Modeling (MFM). The model must reconstruct masked network features from their spatio-temporal context.
- **Goal:** Zero-Day Anomaly Detection. High reconstruction error at inference time indicates a Zero-Day (unseen) attack.

## 3. Technical Constraints
- **Framework:** PyTorch.
- **Hardware:** Optimized for 6GB VRAM (Batch size/Gradient accumulation management).
- **Data Loading:** Must use `DataLoader` with `pin_memory=True` and sharded `.arrow` ingestion to prevent OOM.
- **Feature Engineering:** Maintain 32-bit float precision for continuous flow metrics.

## 4. Coding Standards
- Professional, modular, and type-hinted Python code.
- Implement 'Pre-Norm' LayerNorm configuration for stable foundation model training.
- Use Scaled Dot-Product Attention: $$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$