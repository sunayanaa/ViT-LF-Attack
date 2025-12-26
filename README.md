# ViT-LF-Attack
Official implementation of a wavelet-based probing framework to analyze and visualize the low-frequency adversarial vulnerability of Vision Transformers.
\# On the Low-Frequency Vulnerability of Vision Transformers: A Wavelet-Based Analysis



This repository contains the official PyTorch implementation and reproduction scripts for the paper: \*\*"On the Low-Frequency Vulnerability of Vision Transformers: A Wavelet-Based Analysis"\*\*.



\## 📄 Abstract

We identify a critical reliability gap in Vision Transformers (ViTs). Unlike CNNs, which rely on high-frequency texture, ViTs are vulnerable to adversarial perturbations concentrated in the \*\*Low-Frequency (LL)\*\* band. This repository provides the code to reproduce the spectral analysis, the "Nuclear" latent ablation defense, and the energy distribution experiments reported in the manuscript.



\## 🛠️ Dependencies

Install the required libraries using:

```bash

pip install -r requirements.txt



To reproduce Table I (Defense Failure Rates):

python analysis\_4\_table\_benchmark.py



To reproduce Figure 2 (Spectral Visualization):

python spectral\_visualization.py





