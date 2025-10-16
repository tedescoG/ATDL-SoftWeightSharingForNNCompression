# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository implements "Soft Weight-Sharing for Neural Network Compression" (ICLR 2017) by Karen Ullrich, Edward Meeds, and Max Welling. The approach uses an empirical Bayesian prior (specifically a Gaussian Mixture Model) to simultaneously prune and quantize neural network weights in a single differentiable retraining procedure.

## Architecture

### Core Modules

**empirical_priors.py** - Contains `GaussianMixturePrior`, a custom Keras Layer that implements the GMM prior for weight compression:
- Learns mixture components with trainable means, variances (in log-space as gammas), and mixing proportions (rhos)
- The zero component has fixed mean (μ₀=0) and fixed mixing proportion (π₀, typically 0.99)
- Uses Gamma hyper-priors on the precision parameters to regularize variance estimation
- Returns the negative log-likelihood of weights under the mixture model as the complexity loss

**optimizers.py** - Custom Adam optimizer that supports different learning rates for different parameter types:
- Parameters can be tagged by name (e.g., 'means', 'gammas', 'rhos')
- Each parameter type gets its own learning rate, beta_1, beta_2, and decay schedule
- Essential for training the empirical prior parameters with appropriate step sizes

**extended_keras.py** - Keras engine extensions:
- `extract_weights()`: Collects symbolic trainable weights from a model
- `identity_objective()`: Hack to use a Layer as a loss function for empirical priors
- `logsumexp()`: Numerically stable log-sum-exp implementation for mixture model computations
- `VisualisationCallback`: Generates animated GIFs showing weight distribution evolution during retraining

**helpers.py** - Post-processing utilities:
- `special_flatten()` / `reshape_like()`: Flatten and reshape weight lists from model.get_weights()
- `discretesize()`: Post-training quantization that assigns each weight to its most responsible mixture component
- `compute_responsibilies()`: Computes which mixture component is most responsible for each weight
- `merger()`: Merges nearly identical mixture components based on KL-divergence
- `save_histogram()`: Creates weight distribution visualizations

**dataset.py** - MNIST data loader with proper backend handling for Theano/TensorFlow

### Three-Stage Pipeline

1. **Pretrain**: Train a standard neural network (tutorial uses 2-conv + 2-FC on MNIST)
2. **Retrain with GMM Prior**: Add `GaussianMixturePrior` layer, optimize with custom Adam
3. **Post-process**: Use `discretesize()` to quantize weights to mixture component means

## Key Implementation Details

### Loss Function Structure
The model has two outputs during retraining:
- `error_loss`: Standard categorical cross-entropy for classification
- `complexity_loss`: The GMM prior loss from `GaussianMixturePrior` layer (passed through with `identity_objective`)

Loss weighting: `error_loss` weight = 1.0, `complexity_loss` weight = τ/N where τ is a hyperparameter (~0.003) and N is training set size.

### Parameter Naming Convention
The custom Adam optimizer uses parameter names to assign different learning rates:
- Unnamed parameters: Base learning rate (e.g., 5e-4)
- Parameters with 'means' in name: Learning rate for mixture means (e.g., 1e-4)
- Parameters with 'gammas' in name: Learning rate for log-precisions (e.g., 3e-3)
- Parameters with 'rhos' in name: Learning rate for log-mixing-proportions (e.g., 3e-3)

### Initialization Strategy
- Mixture means: Linearly spaced from -0.6 to 0.6 (excluding zero component)
- Variances: Initialize to 0.25 for all components
- Mixing proportions: Uniform distribution over non-zero components, with π₀ = 0.99 for zero component

### Gamma Hyper-priors
Two different priors on precisions:
- Zero component: (α=5e3, β=20e-1) - stronger prior, expects very small variance
- Other components: (α=2.5e2, β=1e-1) - weaker prior, allows wider variances

## Running the Code

This is a Jupyter notebook-based project. To run:

```bash
jupyter notebook tutorial.ipynb
```

The notebook demonstrates the full pipeline on LeNet-300-100 (MNIST).

### Backend Configuration
The code now uses TensorFlow backend exclusively via `K.backend()`. Legacy Theano support code is preserved but won't execute:
- Input layer shape is channels-last for TensorFlow (28, 28, 1)
- Tensor broadcasting in GMM loss computation uses TensorFlow's `tf.expand_dims()`

## Code Modernization (2025)

**This codebase has been updated from Python 2.7/Keras 1.x to Python 3.9+/Keras 2.12+/TensorFlow 2.12+**

### Changes Made:

**Python 3 Compatibility:**
- `xrange` → `range` (helpers.py:43, 73)
- `zip()` returns iterator → wrapped with `list()` where needed (helpers.py:101)
- Fixed tab/space mixing (helpers.py:136)

**Keras 2.12 API Updates:**
- `from keras.engine.topology import Layer` → `from keras.layers import Layer` (empirical_priors.py:12)
- `K._BACKEND` → `K.backend()` (empirical_priors.py:17, 100; dataset.py:30)
- `K.get_variable_shape()` → `K.int_shape()` (optimizers.py:90)
- `K.variable()` + `self.trainable_weights = [...]` → `self.add_weight()` (empirical_priors.py:36-61)
  - In Keras 2.12, `trainable_weights` is read-only, must use `add_weight()` API
- Custom Layer output shape: `get_output_shape_for()` → `compute_output_shape()` (empirical_priors.py:110-116)
- Custom Layer call must return proper batch dimensions (empirical_priors.py:63-96)
  - Changed from scalar to `(batch_size, 1)` tensor using `K.tile()` and `K.reshape()`
  - Changed `K.variable([0.])` → `K.constant([0.])` for static constants
- Custom Optimizer `get_updates()` signature changed (optimizers.py:75)
  - Old: `get_updates(self, params, constraints, loss)`
  - New: `get_updates(self, loss, params)` - constraints removed, parameter order changed
- Multi-output model metrics must be per-output dict: `metrics={'output1': [...], 'output2': []}` (tutorial.ipynb)
- `Convolution2D` → `Conv2D` with tuple kernel size (tutorial.ipynb)
- `subsample` parameter → `strides` parameter (tutorial.ipynb)
- `Model(input=..., output=...)` → `Model(inputs=..., outputs=...)` (tutorial.ipynb)
- `nb_epoch` parameter → `epochs` parameter (tutorial.ipynb)
- Fixed `super().get_config` → `super().get_config()` (optimizers.py:123)
- Model save/load must use HDF5 format: `save_model(model, "path.h5")` (tutorial.ipynb)

**Library Updates:**
- `scipy.misc.logsumexp` → `scipy.special.logsumexp` (helpers.py:11)
- Commented out deprecated `keras.utils.generic_utils.get_from_module` (optimizers.py:12)
- Seaborn 0.13+ API updates (extended_keras.py:145-147):
  - `sns.jointplot(x, y, ...)` → `sns.jointplot(x=x, y=y, ...)` - use keyword arguments
  - `size` parameter → `height` parameter
  - `stat_func` parameter removed
  - `sns.plt.title()` → `plt.suptitle()` - seaborn no longer has plt attribute
- ImageIO GIF creation: Added image shape validation and fixed DPI for consistent sizing (extended_keras.py:119-141)

### Behavioral Preservation:
All changes are API compatibility updates only. No algorithms, loss functions, or mathematical operations were modified. The numerical behavior and results remain identical to the original implementation.

### Important Limitation:
The custom Adam optimizer in `optimizers.py` is based on the legacy Keras optimizer API (`keras.optimizers.Optimizer`). While all imports work correctly, **TensorFlow 2.x's eager execution must be disabled** for the optimizer to function:

```python
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
```

Add this at the very beginning of the tutorial notebook (first cell after imports) to enable the custom optimizer. This is a limitation of the legacy optimizer API and cannot be resolved without a major rewrite of the optimizer to use TensorFlow 2.x's new optimizer interface.

**Important:** When eager execution is disabled, you must save models in HDF5 format (`.h5` extension) instead of the default SavedModel format. The notebook has been updated to use:
```python
keras.models.save_model(model, "./my_pretrained_net.h5")
keras.models.load_model("./my_pretrained_net.h5")
```

## Dependencies

Current environment requirements:
- Python 3.9+
- keras 2.12+ (TensorFlow-integrated)
- tensorflow 2.12+
- numpy 1.23+
- scipy 1.13+ (logsumexp in scipy.special)
- matplotlib 3.9+
- seaborn 0.13+
- imageio 2.37+
- IPython (for notebook display)

Backend: TensorFlow only (Theano no longer supported)

## Expected Results

On LeNet-300-100 (642K parameters on MNIST):
- Pretrained accuracy: ~99.1%
- Retrained accuracy: ~99.0%
- Post-processed accuracy: ~99.0%
- Compression: ~7.5% non-zero weights (93% pruning)
- Quantization: 16 cluster means (4-bit indices)

## Important Notes

- File paths for temporary visualizations are hardcoded (e.g., `./.tmp%d.png`)
- The GIF generation in `VisualisationCallback` creates and deletes temporary files in the current directory
- The custom optimizer in optimizers.py uses parameter name matching to assign different learning rates