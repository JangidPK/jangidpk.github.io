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

## Mathematical Background
### Anomalous Diffusion

Many physical, biological, and financial systems show diffusion processes that deviate from the classical Brownian motion. Such processes are collectively known as **anomalous diffusion**, where the mean squared displacement (MSD) no longer grows linearly with time.

For a stochastic trajectory $$X(t)$$ , the MSD is defined as
$$\mathrm{MSD} (\tau) = \mathbb{E} \left[(X(t+\tau)-X(t))^2\right]. $$

For normal diffusion,

$$
\mathrm{MSD}(\tau)\propto \tau,
$$

whereas anomalous diffusion follows the power law

$$\mathrm{MSD}(\tau) \propto \tau^{\alpha}$$

where

* $$\alpha<1$$: subdiffusion,
* $$\alpha=1$$: normal diffusion,
* $$\alpha>1$$: superdiffusion.

One of the most important stochastic models capable of describing anomalous diffusion is **Fractional Brownian Motion (fBm)**.

---

### Fractional Brownian Motion

Fractional Brownian Motion (fBm), introduced by Mandelbrot and Van Ness, is a continuous-time Gaussian process
$$
{B_H(t)}_{t\ge0}
$$
parameterized by the **Hurst exponent**, $$H\in(0,1)$$. The process satisfies $$ B_H(0)=0$$, 
and  $$\mathbb{E}[B_H(t)] = 0.$$ Its covariance function is

$$
\operatorname{Cov}(B_H(t),B_H(s))
= \frac12 \left( t^{2H} + s^{2H} - |t-s|^{2H} \right).
$$

Unlike standard Brownian motion, whose increments are independent, fractional Brownian motion possesses correlated increments whose correlation structure is completely determined by the Hurst exponent.

---

### Self-Similarity

Fractional Brownian motion is a self-similar stochastic process satisfying

$$ 
B_H(at)
\overset{d}{=}
a^H B_H(t),
$$

where "$$\overset{d}{=}$$" denotes equality in distribution. Consequently,
 $$ \operatorname{Var}(B_H(t)) = t^{2H}.$$  The parameter $$H$$, therefore,
  determines the scaling behavior of the trajectory.

---

### Mean Squared Displacement

For fBm,

$$
\mathrm{MSD}(\tau) = \mathbb{E}
\left[ (B_H(t+\tau)-B_H(t))^2 \right]
= \tau^{2H}.$$

Comparing with the anomalous diffusion law, $$\mathrm{MSD}(\tau) \propto \tau^\alpha$$, the anomalous diffusion exponent is
$$
\alpha=2H.
$$
Therefore,

* $$H<0.5$$ corresponds to subdiffusion. Here positive increments are likely to be followed by negative increments, producing highly irregular trajectories with frequent direction reversals.
* $$H=0.5$$ corresponds to classical Brownian motion, where successive increments are independent.


* $$H>0.5$$ corresponds to superdiffusion, where successive increments tend to preserve their direction, resulting in smoother trajectories exhibiting long-memory behavior.

---

### Fractional Gaussian Noise

The increment process
$$ X_k = B_H(k)-B_H(k-1)$$ is known as **Fractional Gaussian Noise (fGn)**. Unlike fBm, fGn is stationary. Its autocovariance is 

$$
\gamma(k) = \frac{1}{2} \left( |k-1|^{2H} - 2|k|^{2H} + |k+1|^{2H} \right) $$.


For large lags, $$\gamma(k) \sim H(2H-1)k^{2H-2}$$. This slow algebraic decay gives rise to long-range correlations.

---

### Statistical Estimation Problem

Suppose $$
\mathcal{D} = \{ (\mathbf{x}_i,H_i) \}_{i=1}^{N}
$$ denotes a dataset of simulated trajectories. Each trajectory is represented as $$\mathbf{x}_i = (x_{i1},x_{i2},\ldots,x_{iT}) \in  \mathbb{R}^{T}$$, while $$H_i\in(0,1)$$ is the corresponding Hurst exponent.The objective is to learn a nonlinear mapping
$$f_\theta: \mathbb{R}^{T} \rightarrow (0,1) $$ parameterized by $$\theta$$ such that  $$\hat H_i  = f_\theta(\mathbf{x}_i) $$ approximates the true Hurst exponent (note this is a standard process in machine learning).

---

### Learning Objective

The model parameters are estimated by minimizing the empirical mean squared error 

$$\mathcal{L}(\theta) = \frac1N \sum_{i=1}^{N} \left(
H_i - f_\theta(\mathbf{x}_i) \right)^2.$$

The optimization problem is therefore
$$ \theta^\ast =\arg\min_\theta \mathcal{L}(\theta).$$
After training, the estimated Hurst exponent $$\hat H=f_{\theta^\ast}(\mathbf{x})$$, characterizes the anomalous diffusion regime of an unseen trajectory.


Unlike traditional Hurst estimation techniques, which rely on handcrafted statistics such as rescaled range analysis, detrended fluctuation analysis, or spectral methods, the objective here is to infer the Hurst exponent directly from raw trajectories by learning the underlying temporal correlation structure. This allows the model to approximate the inverse mapping from observed sample paths to the latent parameter $H$, enabling data-driven estimation of anomalous diffusion across anti-persistent, Brownian, and persistent regimes.


## Tools
Python, PyTorch, NumPy, SciPy, scikit-learn, Matplotlib, pandas,


## Applications

* Anomalous diffusion analysis
* Time-series parameter estimation
* Statistical physics
* Quantitative finance
* Signal processing
* Scientific machine learning

## Repository

This project demonstrates how modern sequence models can infer the Hurst exponent from stochastic trajectories, providing an efficient alternative to traditional statistical estimators.

[GitHub Code](https://github.com/JangidPK/fbm-hurst-estimation)
