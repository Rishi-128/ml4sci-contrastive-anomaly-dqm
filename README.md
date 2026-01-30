# Anomaly Detection for Data Quality Monitoring using Contrastive Learning

This project explores anomaly detection techniques for detector data quality monitoring (DQM) using representation learning.  
The goal is to learn robust feature embeddings from normal detector behavior and use them to identify anomalous patterns.

The project is inspired by real-world challenges in high-energy physics experiments, where detector conditions evolve over time and labeled anomaly data is limited.

---

## Project Overview

We follow a three-stage pipeline:

1. **Synthetic Detector Data Generation**
   - Simulate normal detector behavior using multivariate signals
   - Generate anomalous samples representing abnormal detector conditions

2. **Contrastive Pretraining**
   - Apply data augmentations such as noise injection and scaling
   - Train a contrastive encoder to learn meaningful latent representations
   - Optimize representations without requiring anomaly labels

3. **Anomaly Detection**
   - Use reconstruction error in latent space for anomaly scoring
   - Compare error distributions for normal vs anomalous data
   - Visualize separation between normal and anomalous samples

---

## Repository Structure
```
├── 01_data_generation.ipynb # Synthetic detector data simulation
├── 02_contrastive_pretraining.ipynb # Contrastive representation learning
├── 03_anomaly_detection.ipynb # Anomaly detection using learned embeddings
└── README.md
```

---

## Key Results

- Contrastive learning produces stable latent representations of normal detector behavior
- Anomalous samples show significantly higher reconstruction error
- Clear separation is observed between normal and anomalous error distributions

---

## Technologies Used

- Python
- NumPy
- PyTorch
- Matplotlib
- Google Colab

---

## Motivation

In realistic DQM scenarios, anomalies are rare and detector conditions evolve continuously.  
Contrastive learning enables learning robust representations without explicit anomaly labels, making it well-suited for scientific monitoring tasks.

---

## Future Extensions

- Online / continuous learning for evolving detector conditions
- Integration with autoencoder-based reconstruction models
- Evaluation on real experimental datasets

---

## Author

Rishi
(GSoC 2026 preparation project – ML4SCI)
