---
title: "PCA-Based Equity Research and Signal Generation"
excerpt: "Developed a quantitative equity research pipeline using Principal Component Analysis (PCA) for factor extraction, residual analysis, and systematic long-short signal generation in Indian equity markets.<br/><img src='/images/pc_ratio.png' style='max-width:600px; width:100%;'>"
collection: portfolio
---


This project explores the application of statistical factor models to equity markets using Principal Component Analysis (PCA). The objective was to identify latent risk factors driving stock returns and exploit residual mispricings to generate systematic trading signals.

## Overview

The workflow consists of:

1. Constructing a return matrix from historical stock price data.
2. Extracting dominant latent factors using PCA.
3. Reconstructing stock returns using a reduced factor representation.
4. Computing residual returns as deviations from factor-implied behavior.
5. Generating long-short trading signals based on residual mispricing.
6. Evaluating portfolio performance through backtesting.

## Methodology

### Principal Component Analysis

PCA is used to identify common sources of variation across stocks and reduce the dimensionality of the return space. The leading principal components capture broad market and sector-level movements, while residual returns represent stock-specific deviations.

### Signal Generation

Stocks with large positive residuals are interpreted as potentially overvalued relative to the factor model, while stocks with large negative residuals are interpreted as potentially undervalued. These residuals are used to construct market-neutral long-short portfolios.

### Backtesting

Portfolio performance is evaluated using historical data through rolling-window analysis. Key performance metrics include cumulative returns, volatility, Sharpe ratio, and drawdown statistics.

## Tools

- Python
- NumPy
- pandas
- scikit-learn
- Matplotlib
- Principal Component Analysis (PCA)
- Quantitative Finance and Factor Modeling

## Key Outcomes

- Developed a complete PCA-based equity research workflow.
- Implemented rolling factor estimation and residual analysis.
- Constructed systematic long-short trading signals.
- Performed out-of-sample backtesting and performance evaluation.
- Demonstrated applications of statistical learning methods in quantitative finance.

## Repository

GitHub Repository:
[Link](https://github.com/JangidPK/PCA-factor-modeling-and-signal-generation)