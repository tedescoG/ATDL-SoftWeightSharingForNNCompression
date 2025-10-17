# Soft Weight-Sharing for Neural Network Compression

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Keras 2.12+](https://img.shields.io/badge/keras-2.12+-red.svg)](https://keras.io/)
[![TensorFlow 2.12+](https://img.shields.io/badge/tensorflow-2.12+-orange.svg)](https://www.tensorflow.org/)

This repository contains a **modernized implementation** of "Soft Weight-Sharing for Neural Network Compression" by Ullrich et al. (ICLR 2017). The approach uses an empirical Bayesian prior (specifically a Gaussian Mixture Model) to simultaneously prune and quantize neural network weights in a single differentiable retraining procedure.

## 📋 Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Modernization Changes](#modernization-changes)
- [Installation](#installation)
- [Usage](#usage)
- [Architecture](#architecture)
- [Results](#results)
- [References](#references)

## 🎯 Overview

Traditional neural network compression approaches (e.g., Han et al. 2016) use a multi-stage pipeline with separate pruning and quantization steps. In contrast, this implementation:

- **Simultaneously prunes and quantizes** weights in one differentiable procedure
- Uses a **Gaussian Mixture Model (GMM) prior** for weight compression
- Implements **empirical Bayesian learning** where prior parameters are learned alongside network weights
- Achieves **~93% pruning** with **4-bit quantization** while maintaining accuracy

### Three-Stage Pipeline

1. **Pretrain**: Train a standard neural network (e.g., 2-conv + 2-FC on MNIST)
2. **Retrain with GMM Prior**: Add `GaussianMixturePrior` layer, optimize with custom Adam
3. **Post-process**: Use quantization to assign weights to mixture component means

## ✨ Key Features

- **Single differentiable procedure** for compression (vs. multi-stage pipelines)
- **Custom Adam optimizer** with parameter-specific learning rates
- **Gamma hyper-priors** on variance parameters for regularization
- **Visualization callbacks** showing weight distribution evolution during training
- **Post-processing utilities** for weight quantization and merging

## 🔧 Modernization Changes

> **⚠️ Important**: This codebase has been updated from **Python 2.7/Keras 1.x** to **Python 3.9+/Keras 2.12+/TensorFlow 2.12+**

### Python 3 Compatibility

- ✅ `xrange` → `range` (helpers.py:43, 73)
- ✅ `zip()` returns iterator → wrapped with `list()` where needed (helpers.py:101)
- ✅ Fixed tab/space mixing issues (helpers.py:136)

### Keras 2.12 API Updates

**Layer Implementation:**
- ✅ `from keras.engine.topology import Layer` → `from keras.layers import Layer` (empirical_priors.py:12)
- ✅ `K._BACKEND` → `K.backend()` (empirical_priors.py:17, 100; dataset.py:30)
- ✅ `K.get_variable_shape()` → `K.int_shape()` (optimizers.py:90)
- ✅ `K.variable()` + manual `trainable_weights` → `self.add_weight()` API (empirical_priors.py:36-61)
- ✅ `get_output_shape_for()` → `compute_output_shape()` (empirical_priors.py:110-116)

**Model API:**
- ✅ `Convolution2D` → `Conv2D` with tuple kernel size
- ✅ `subsample` parameter → `strides` parameter
- ✅ `Model(input=..., output=...)` → `Model(inputs=..., outputs=...)`
- ✅ `nb_epoch` → `epochs` parameter
- ✅ Multi-output metrics: `metrics=['accuracy']` → `metrics={'output1': [...], 'output2': [...]}`

**Optimizer:**
- ✅ `get_updates(self, params, constraints, loss)` → `get_updates(self, loss, params)` (optimizers.py:75)
- ✅ Fixed `super().get_config` → `super().get_config()` (optimizers.py:123)

**Save/Load:**
- ✅ Model save/load uses HDF5 format: `save_model(model, "path.h5")` (required with eager execution disabled)

### Library Updates

- ✅ `scipy.misc.logsumexp` → `scipy.special.logsumexp` (helpers.py:11)
- ✅ Seaborn 0.13+ API updates (extended_keras.py:145-147):
  - `sns.jointplot(x, y, ...)` → `sns.jointplot(x=x, y=y, ...)`
  - `size` parameter → `height` parameter
  - `stat_func` parameter removed
  - `sns.plt.title()` → `plt.suptitle()`
- ✅ ImageIO GIF creation: Added image shape validation and fixed DPI (extended_keras.py:119-141)

### Critical Limitation: Eager Execution

**⚠️ The custom Adam optimizer requires disabling TensorFlow's eager execution:**

```python
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
```

This must be added at the **very beginning** of the tutorial notebook (first cell after imports). This is a limitation of the legacy Keras optimizer API and cannot be resolved without a major rewrite.

**Important:** When eager execution is disabled, models must be saved in HDF5 format (`.h5` extension) instead of the default SavedModel format.

### Behavioral Preservation

✅ **All changes are API compatibility updates only**
✅ **No algorithms, loss functions, or mathematical operations were modified**
✅ **Numerical behavior and results remain identical to the original implementation**

## 📦 Installation

### Prerequisites

- Python 3.9 or higher
- pip package manager

### Step 1: Clone the Repository

```bash
git clone <repository-url>
cd keras-ft-SWS
```

### Step 2: Create Virtual Environment (Recommended)

```bash
# Using venv
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# OR using conda
conda create -n sws python=3.9
conda activate sws
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Verify Installation

```bash
python -c "import tensorflow as tf; import keras; print('TensorFlow:', tf.__version__); print('Keras:', keras.__version__)"
```

Expected output:
```
TensorFlow: 2.12.x
Keras: 2.12.x
```

## 🚀 Usage

### Running the Tutorial

1. **Start Jupyter Notebook:**

```bash
jupyter notebook tutorial.ipynb
```

2. **Execute the cells in order:**

The notebook demonstrates the complete pipeline on LeNet-300-100 (MNIST):

- **Part 1**: Pretraining a neural network
- **Part 2**: Retraining with empirical GMM prior
- **Part 3**: Post-processing and quantization

3. **Important First Step:**

The **first code cell** must disable eager execution:

```python
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
```

### Expected Runtime

- **Pretraining**: ~2-3 minutes (20 epochs)
- **Retraining with GMM**: ~15-20 minutes (50 epochs)
- **Post-processing**: <1 minute

### Key Configuration Parameters

**GMM Prior Configuration:**
```python
GaussianMixturePrior(
    nb_components=16,           # Number of mixture components (4-bit quantization)
    network_weights=extract_weights(model),
    pretrained_weights=pretrained_model.get_weights(),
    pi_zero=0.99,              # Mixing proportion for zero component
    name="complexity_loss"
)
```

**Custom Optimizer:**
```python
optimizers.Adam(
    lr=[5e-4, 1e-4, 3e-3, 3e-3],  # [unnamed, means, log(precision), log(mixing proportions)]
    param_types_dict=['means', 'gammas', 'rhos']
)
```

**Loss Weighting:**
```python
tau = 0.003
N = X_train.shape[0]
loss_weights = {
    "error_loss": 1.0,
    "complexity_loss": tau/N
}
```

## 🏗️ Architecture

### Core Modules

**`empirical_priors.py`** - GMM prior implementation
- `GaussianMixturePrior`: Custom Keras Layer implementing the GMM prior
- Learns mixture components with trainable means, variances, and mixing proportions
- Zero component has fixed mean (μ₀=0) and fixed mixing proportion (π₀)
- Uses Gamma hyper-priors on precision parameters

**`optimizers.py`** - Custom Adam optimizer
- Supports different learning rates for different parameter types
- Parameters tagged by name ('means', 'gammas', 'rhos')
- Each type gets its own learning rate, beta_1, beta_2, and decay

**`extended_keras.py`** - Keras extensions
- `extract_weights()`: Collects symbolic trainable weights
- `identity_objective()`: Uses a Layer as a loss function
- `logsumexp()`: Numerically stable log-sum-exp
- `VisualisationCallback`: Generates animated GIFs of weight evolution

**`helpers.py`** - Post-processing utilities
- `discretesize()`: Post-training quantization
- `compute_responsibilities()`: Component responsibility calculation
- `merger()`: Merges similar mixture components
- `save_histogram()`: Weight distribution visualization

**`dataset.py`** - MNIST data loader with backend handling

## 📊 Results

### Expected Performance on LeNet-300-100 (MNIST, 642K parameters)

| Metric | Value |
|--------|-------|
| **Pretrained Accuracy** | ~98.9% |
| **Retrained Accuracy** | ~99.0% |
| **Post-processed Accuracy** | ~99.0% |
| **Compression Rate** | ~7.5% non-zero weights (93% pruning) |
| **Quantization** | 16 cluster means (4-bit indices) |
| **Effective Compression** | ~30x size reduction |

### Visualization Outputs

The training process generates:
- `figures/retraining.gif` - Animated weight distribution evolution
- `figures/reference.png` - Pretrained weight histogram
- `figures/retrained.png` - Retrained weight histogram
- `figures/post-processed.png` - Final quantized weights

## 🌍 Environment Snapshot

For reproducibility, here's the complete environment used for testing:

### Core Dependencies

**Deep Learning Framework:**
```
tensorflow==2.12.0
  - Backend: CPU/GPU support
  - Includes: tf.keras integration
  - Important: Eager execution must be disabled

keras==2.12.0
  - Integrated with TensorFlow 2.12
  - Uses legacy optimizer API for custom Adam
```

**Numerical Computing:**
```
numpy==1.23.5
  - Core numerical operations
  - Compatible with TensorFlow 2.12

scipy==1.13.0
  - Used for: scipy.special.logsumexp
  - Replaces deprecated scipy.misc.logsumexp
```

### Visualization Stack

```
matplotlib==3.9.0
  - Plot generation for weight histograms
  - Figure creation for GIF animations

seaborn==0.13.0
  - Statistical visualizations
  - Updated API: keyword arguments (x=, y=, height=)

imageio==2.37.0
  - GIF creation from frame sequences
  - Used in VisualisationCallback
```

### Jupyter Environment

```
jupyter==1.0.0
notebook==6.5.4
ipython==8.12.0
ipykernel==6.22.0
```

### System Information

```
Platform: macOS Darwin 25.0.0 (also compatible with Linux/Windows)
Python: CPython 3.9+
Architecture: x86_64 / ARM64 (Apple Silicon compatible)
```

### Installation Methods

**Method 1: pip (recommended)**
```bash
python3.9 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**Method 2: conda**
```bash
conda env create -f environment.yml
conda activate sws
```

### Compatibility Notes

**TensorFlow 2.12 Limitations:**
1. **Eager Execution**: Must be disabled for custom optimizer
   ```python
   tf.compat.v1.disable_eager_execution()
   ```
2. **Model Saving**: Must use HDF5 format (`.h5`) when eager execution disabled
3. **Legacy Optimizer API**: Cannot easily upgrade without major rewrite

**Platform-Specific:**
- **macOS (M1/M2/M3)**: May require Rosetta 2 or use `tensorflow-macos`
- **Linux**: CUDA support available for GPU acceleration
- **Windows**: CPU-only or requires WSL2 for GPU support

### Tested Environment

```
Date: October 2025
OS: macOS Darwin 25.0.0
Python: 3.9.13
TensorFlow: 2.12.0
Keras: 2.12.0
```

All tests passed achieving:
- Pretrained: ~98.9% accuracy
- Retrained: ~99.0% accuracy
- Post-processed: ~99.0% accuracy
- Compression: 93% pruning (7.5% non-zero weights)
- Quantization: 16 clusters (4-bit)

See `requirements.txt` for exact versions and `environment.yml` for conda environment specification.

## 📚 References

### Original Paper

```bibtex
@inproceedings{ullrich2017soft,
    title={Soft Weight-Sharing for Neural Network Compression},
    author={Ullrich, Karen and Meeds, Edward and Welling, Max},
    booktitle={ICLR 2017},
    year={2017}
}
```

### Related Work

- Han et al. (2016) - "Deep Compression: Compressing Deep Neural Networks with Pruning, Trained Quantization and Huffman Coding"
- Original implementation: [GitHub - KarenUllrich/Tutorial_BayesianCompressionForDL](https://github.com/KarenUllrich/Tutorial_BayesianCompressionForDL)

## 📝 Notes

- Temporary visualization files are created in the current directory (`.tmp%d.png`)
- GIF generation creates and deletes temporary files during training
- The custom optimizer uses parameter name matching for learning rate assignment
- File paths in the notebook may need adjustment based on your directory structure

## 🤝 Contributing

This is a modernization of the original codebase for educational and research purposes. Contributions to further improve compatibility or fix issues are welcome.

## 📄 License

Please refer to the original repository for licensing information.

---

**Modernized by**: Gaetano Tedesco  
**Original Authors**: Karen Ullrich, Edward Meeds, Max Welling  
**Institution**: University of Copenhagen (UCPH) - Advanced Topics in Deep Learning (ATDL)