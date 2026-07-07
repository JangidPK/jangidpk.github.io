---
title: "Anomalous Hurst Coefficient Prediction using Sequence Models"
excerpt: "A deep learning framework for estimating the Hurst exponent of Fractional
 Brownian Motion trajectories using LSTM and Transformer models.
 <br/><img src='/images/fgn.png' style='max-width:600px; width:100%;'>"
collection: portfolio
---

This project investigates the estimation of the **Hurst exponent (H)** from simulated Fractional Brownian Motion (fBm) trajectories using deep learning. It implements exact fBm simulation algorithms, classical machine learning baselines, and sequence models including Bidirectional LSTMs and Transformers to learn long-range temporal dependencies directly from raw trajectories. The framework provides an end-to-end pipeline for data generation, model training, evaluation, and visualization. :contentReference[oaicite:0]{index=0}

## Features

* Exact fBm simulation (Davies–Harte, Hosking, Cholesky)
* Bidirectional LSTM and Transformer regressors
* Random Forest baseline using handcrafted statistical features
* Comprehensive evaluation and visualization tools
* PyTorch-based training.

## Tools

* Python
* PyTorch
* NumPy
* SciPy
* scikit-learn
* Matplotlib
* pandas

## Applications

* Anomalous diffusion analysis
* Time-series parameter estimation
* Statistical physics
* Quantitative finance
* Signal processing
* Scientific machine learning

## Repository

This project demonstrates how modern sequence models can accurately infer the Hurst exponent from stochastic trajectories, providing an efficient alternative to traditional statistical estimators.

**GitHub Repository:**  
[Link](https://github.com/JangidPK/fbm-hurst-estimation)
