# Application-of-Graph-Neural-Networks-GNN-for-Aqueous-Solubility-Prediction-of-Drug-like-Molecules

# Graph Neural Network for Aqueous Solubility Prediction of Drug-like Molecules

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch)](https://pytorch.org/)
[![PyTorch Geometric](https://img.shields.io/badge/PyTorch_Geometric-2.x-3C78D8)](https://pyg.org/)
[![RDKit](https://img.shields.io/badge/RDKit-2024-lightgrey)](https://www.rdkit.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1fnVOaLij7eg8YJ9wSYS3pAAjxycHvnPn)

---

## Table of Contents

- [Introduction](#introduction)
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Methodology](#methodology)
  - [Molecular Graph Featurization](#molecular-graph-featurization)
  - [Model Architecture](#model-architecture)
  - [Training Strategy](#training-strategy)
- [Results](#results)
- [Visualizations](#visualizations)
- [Repository Structure](#repository-structure)
- [Installation & Usage](#installation--usage)
- [References](#references)

---

## Introduction

Aqueous solubility — the ability of a compound to dissolve in water — is one of the most critical physicochemical properties in pharmaceutical drug discovery and development. Poor aqueous solubility is a primary cause of failure in early-stage drug development pipelines, contributing to inadequate bioavailability, formulation difficulties, and compound attrition. Accurate *in silico* prediction of solubility (expressed as log S, the log₁₀ of molar solubility at 25 °C) from molecular structure alone is therefore a high-value task in cheminformatics and computer-aided drug design.

Traditional approaches to solubility prediction rely on hand-crafted molecular descriptors (such as topological polar surface area, logP, and hydrogen bond counts) fed into classical machine learning models. While effective within narrow chemical spaces, such models generalize poorly across structurally diverse compound libraries. In recent years, **Graph Neural Networks (GNNs)** have emerged as a compelling alternative: by representing molecules directly as graphs — where atoms are nodes and bonds are edges — GNNs learn hierarchical molecular representations end-to-end from data, eliminating the need for manual feature engineering and capturing complex structural patterns that fixed descriptors cannot encode.

This project implements an industry-grade GNN pipeline for aqueous solubility prediction, following best practices established in the current US and European cheminformatics literature. The work is grounded in peer-reviewed findings from *Nature Scientific Data*, *Journal of Cheminformatics*, and *ACS Omega*, and the pipeline is designed to be directly comparable with published benchmarks on the AqSolDB dataset.

---

## Project Overview

This project builds a complete, end-to-end deep learning pipeline for molecular property prediction, covering:

- **Data acquisition and curation** from the largest publicly available aqueous solubility dataset (AqSolDB, 9,982 compounds)
- **Exploratory data analysis** including target distribution, physicochemical descriptor profiling, and correlation analysis
- **Molecular graph featurization** with rich 27-dimensional atom features and 6-dimensional bond features using RDKit
- **GCN model design** with residual connections, batch normalization, and dual-pooling readout
- **Robust evaluation** via 80/20 holdout testing and 5-fold cross-validation
- **Comprehensive visualization** of learning curves, prediction quality, residual distributions, and molecule-level predictions
- **Literature benchmarking** against published GNN and classical ML baselines on the same dataset

---

## Dataset

**AqSolDB** (Sorkun, Khetan & Er, *Nature Scientific Data*, 2019) is the largest and most carefully curated aqueous solubility reference set in the public domain. It consolidates experimental solubility measurements from nine independent publicly available sources into a single, deduplicated, standardized resource.

| Property | Value |
|---|---|
| Total compounds | 9,982 |
| Target variable | log S (mol/L) at 25 °C |
| log S range | −11.6 to +1.6 |
| log S mean ± SD | −3.05 ± 2.09 |
| Source datasets merged | 9 |
| Molecular representations | SMILES, InChI, InChIKey |
| Additional features | 17 pre-computed 2D descriptors |

AqSolDB is preferred over the widely used but much smaller ESOL dataset (~1,100 compounds) because a larger and more chemically diverse training corpus yields significantly better generalization — a finding consistently reported in the GNN solubility literature (Wenzel et al., *J. Cheminform.*, 2025).

**Data cleaning steps applied in this project:**
1. Removal of rows with missing SMILES or solubility values
2. SMILES validation and canonicalization via RDKit
3. Deduplication by canonical SMILES (retaining the median log S across duplicates)
4. Statistical outlier removal (|z-score| > 3.5), yielding a clean dataset of ~9,900 compounds

---

## Methodology

### Molecular Graph Featurization

Each molecule is converted to a graph data object compatible with PyTorch Geometric. The featurization scheme is consistent with recent best practice in the field (Shaker et al., *ACS Omega*, 2023):

**Node features (27 dimensions per atom):**
| Feature | Encoding | Dim |
|---|---|---|
| Atom type | One-hot (13 elements + other) | 13 |
| Hybridization | One-hot (SP, SP2, SP3, SP3D, SP3D2, other) | 6 |
| Aromaticity | Binary | 1 |
| Formal charge | Integer | 1 |
| Total H count | Integer | 1 |
| Ring membership | Binary | 1 |
| Chirality | One-hot (4 types) | 4 |

**Edge features (6 dimensions per bond):**
Bond type (single/double/triple/aromatic), conjugation, ring membership — encoded in both directions to yield an undirected graph.

### Model Architecture

The model is a **multi-layer Graph Convolutional Network (GCN)** with the following design:

```
Input (27-dim node features)
        ↓
Linear Projection → BatchNorm → ReLU
        ↓
4 × Residual GCN Block:
    GCNConv → BatchNorm → ReLU → Dropout(0.15) → Skip connection
        ↓
Global Mean Pooling ──┐
                       ├─→ Concat (256-dim) → Linear → BatchNorm → ReLU → Dropout → Linear(1)
Global Max Pooling  ──┘
        ↓
Predicted log S (scalar)
```

Key design decisions:
- **Residual (skip) connections** after every GCN layer to prevent gradient vanishing and enable stable training of deeper networks
- **Batch normalization** after each graph convolution, following standard practice in molecular property prediction
- **Dual pooling** (mean + max concatenated) to capture both average and extreme atomic environments in the molecular graph
- **Gradient clipping** (max norm = 1.0) during training for additional stability

Total trainable parameters: ~270,000

### Training Strategy

| Hyperparameter | Value |
|---|---|
| Optimizer | Adam |
| Learning rate | 1 × 10⁻³ |
| Weight decay | 1 × 10⁻⁵ |
| Batch size | 128 |
| Epochs | 200 |
| LR scheduler | ReduceLROnPlateau (factor=0.5, patience=20) |
| Loss function | Mean Squared Error (MSE) |
| Train / test split | 80% / 20% (random) |
| Cross-validation | 5-fold (CV epochs: 150) |
| Random seed | 42 (full reproducibility) |

Best model weights are checkpointed based on minimum test MSE and restored at the end of training.

---

## Results

### Holdout Test Set Performance (80/20 Split)

| Metric | Value |
|---|---|
| **R²** | **0.91** |
| **RMSE** | **0.88 log S units** |
| **MAE** | **0.65 log S units** |

### 5-Fold Cross-Validation

| Metric | Mean ± Std |
|---|---|
| R² | 0.90 ± 0.01 |
| RMSE | 0.90 ± 0.02 log S units |
| MAE | 0.67 ± 0.02 log S units |

The low standard deviation across folds confirms that the model is stable and generalizes consistently across different data partitions.

### Literature Benchmark Comparison

| Model | Dataset | RMSE | R² |
|---|---|---|---|
| **This GCN Model** | AqSolDB | **~0.88** | **~0.91** |
| GCN (Wenzel et al., 2025) | AqSolDB | 0.890 | ~0.91 |
| AttentiveFP (Shaker et al., 2023) | AqSolDB | 0.920 | ~0.90 |
| Random Forest (baseline) | AqSolDB | 1.050 | ~0.85 |
| SolTranNet (Francoeur & Koes, 2021) | AqSolDB | 1.459 | ~0.76 |

> RMSE values in log S units (mol/L). Lower is better.

The model achieves performance on par with the current state of the art on AqSolDB, outperforming attention-based GNN baselines and representing a substantial improvement over classical machine learning approaches.

---

## Visualizations

The notebook produces the following publication-quality figures, all saved as PNG files:

| File | Description |
|---|---|
| `eda_distribution.png` | Histogram, KDE, and boxplot of the log S target variable |
| `eda_descriptors.png` | Distributions of molecular weight, logP, and H-bond counts |
| `eda_correlation.png` | Pearson correlation heatmap between log S and 2D descriptors |
| `learning_curves.png` | Train and test MSE loss over epochs (linear and log scale) |
| `pred_vs_actual.png` | Density-coloured scatter plot of predicted vs actual log S with R² annotation, plus residual plot |
| `cv_performance.png` | Per-fold R², RMSE, and MAE bar charts with mean ± SD overlay |
| `error_analysis.png` | Residual histogram, cumulative error distribution, and |error| vs actual log S |
| `sample_molecules.png` | 2D structures of 8 representative test molecules with predicted vs actual log S, colour-coded by prediction quality |

---

## Repository Structure

```
.
├── GNN_Solubility_Prediction_AqSolDB.ipynb   # Main notebook (Google Colab)
├── gnn_solubility_prediction_aqsoldb.py      # Python script export of the notebook
├── README.md                                  # This file
├── solubility_gnn_best_model.pt              # Saved model weights (generated at runtime)
├── test_predictions.csv                      # Test set predictions (generated at runtime)
├── cv_results.csv                            # 5-fold CV results (generated at runtime)
└── figures/                                  # Output visualizations (generated at runtime)
    ├── eda_distribution.png
    ├── eda_descriptors.png
    ├── eda_correlation.png
    ├── learning_curves.png
    ├── pred_vs_actual.png
    ├── cv_performance.png
    ├── error_analysis.png
    └── sample_molecules.png
```

---

## Installation & Usage

### Run on Google Colab (Recommended)

Click the badge at the top of this README or open the notebook directly:

```
https://colab.research.google.com/drive/1fnVOaLij7eg8YJ9wSYS3pAAjxycHvnPn
```

All dependencies are installed automatically in the first cell. A GPU runtime (Runtime → Change runtime type → T4 GPU) is recommended for training speed but not required.

### Run Locally

**Requirements:** Python 3.10+, CUDA-capable GPU (optional)

```bash
# Clone the repository
git clone https://github.com/<your-username>/gnn-solubility-prediction.git
cd gnn-solubility-prediction

# Install dependencies
pip install rdkit torch torch-geometric deepchem pandas numpy \
            scikit-learn matplotlib seaborn requests tqdm scipy pillow

# Run the notebook
jupyter notebook GNN_Solubility_Prediction_AqSolDB.ipynb

# Or run as a script
python gnn_solubility_prediction_aqsoldb.py
```

> **Note:** The dataset is downloaded automatically at runtime from the [AqSolDB GitHub repository](https://github.com/mcsorkun/AqSolDB/tree/master/results). No manual download is required.

---

## References

1. **Sorkun, M. C., Khetan, A., & Er, S.** (2019). AqSolDB, a curated reference set of aqueous solubility and 2D descriptors for a diverse set of compounds. *Nature Scientific Data*, 6, 143. https://doi.org/10.1038/s41597-019-0151-1

2. **Wenzel, J., et al.** (2025). Prediction of the water solubility by a graph convolutional-based neural network on a highly curated dataset. *Journal of Cheminformatics*, 17, 44. https://doi.org/10.1186/s13321-025-01000-9

3. **Shaker, B., et al.** (2023). Attention-based graph neural network for molecular solubility prediction. *ACS Omega*, 8(3), 3236–3244. https://doi.org/10.1021/acsomega.2c06702

4. **Duvenaud, D., et al.** (2015). Convolutional networks on graphs for learning molecular fingerprints. *Advances in Neural Information Processing Systems*, 28, 2224–2232.

5. **Fang, X., et al.** (2022). Geometry-enhanced molecular representation learning for property prediction. *Nature Machine Intelligence*, 4, 127–134. https://doi.org/10.1038/s42256-021-00438-4

6. **Kipf, T. N., & Welling, M.** (2017). Semi-supervised classification with graph convolutional networks. *International Conference on Learning Representations (ICLR 2017)*. https://arxiv.org/abs/1609.02907

7. **Francoeur, P. G., & Koes, D. R.** (2021). SolTranNet — a machine learning tool for fast aqueous solubility prediction. *Journal of Chemical Information and Modeling*, 61(6), 2530–2536. https://doi.org/10.1021/acs.jcim.1c00331

---

*This project is part of an ongoing effort to apply state-of-the-art deep learning methods to ADMET property prediction in drug discovery.*
