# Simulation-Based Inference (SBI) Tutorials
README and Installation instruction generated automatically with Claude Code

Tutorials for the **DESC Stellar Stream Workshop** at University of Washington (November 17-21, 2024)

## Overview

This repository contains hands-on tutorials introducing simulation-based inference methods for astrophysics applications, with a focus on inferring dark matter subhalo properties from stellar streams.

## Installation

See [INSTALL.md](INSTALL.md) for detailed installation instructions.

Quick start:
```bash
pip install -r requirements.txt
```

## Tutorial Contents

### Tutorial 01: Neural Posterior Estimation (NPE)
**File**: [01_neural_posterior_estimation.ipynb](01_neural_posterior_estimation.ipynb)

Introduction to NPE, the most common SBI method. Learn to:
- Directly model the posterior distribution p(θ|x) using neural networks
- Compare Gaussian vs flow-based posteriors
- Understand when normalizing flows are necessary
- Apply NPE to Gaussian and half-moon datasets

**Key takeaway**: Use normalizing flows (not Gaussian models) for flexible posterior estimation.

### Tutorial 02: Neural Likelihood Estimation (NLE)
**File**: [02_neural_likelihood_estimation.ipynb](02_neural_likelihood_estimation.ipynb)

Learn to model the likelihood p(x|θ) instead of the posterior. Topics include:
- Training neural networks to approximate likelihoods
- Combining learned likelihoods with MCMC and Nested Sampling
- Comparing NLE with NPE approaches
- Understanding when to use each method

**Key takeaway**: NLE provides flexibility in choosing sampling algorithms but requires MCMC/Nested Sampling for inference.

### Tutorial 03: Neural Ratio Estimation (NRE)
**File**: [03_neural_ratio_estimation.ipynb](03_neural_ratio_estimation.ipynb)

Learn to use binary classification for inference. This tutorial covers:
- Training classifiers to distinguish joint from marginal samples
- Computing likelihood ratios from classifier outputs
- Applying NRE to complex distributions without flows
- Understanding the advantages of ratio estimation

**Key takeaway**: NRE achieves competitive performance with simpler architectures than NPE/NLE.

### Tutorial 04: Model Validation and Calibration
**File**: [04_model_validation.ipynb](04_model_validation.ipynb)

Essential techniques for validating SBI models. Learn to:
- Perform rank-based calibration tests
- Apply TARP (Tests of Accuracy with Random Points)
- Detect over-confident and under-confident posteriors
- Interpret calibration diagnostics

**Key takeaway**: Calibration tests are critical for ensuring your posteriors are trustworthy.

### Tutorial 05a: Simple NPE for Stellar Streams
**File**: [05a_simpleNPE_streams.ipynb](05a_simpleNPE_streams.ipynb)

Apply NPE to a realistic astrophysics problem: inferring subhalo properties from stellar streams. Topics include:
- Loading and preprocessing stream simulation data
- Building MLP + Neural Spline Flow models
- Validating posteriors with coverage tests
- Understanding data normalization for SBI

**Application**: Infer subhalo mass and impact velocity from perturbed stellar streams.

### Tutorial 05b: Graph NPE for Stellar Streams
**File**: [05b_graphNPE_streams.ipynb](05b_graphNPE_streams.ipynb)

Advanced architecture using Graph Attention Networks (GAT) for stream analysis. Learn to:
- Construct k-nearest neighbor graphs from particle data
- Use Graph Attention Networks to capture spatial structure
- Compare graph-based vs MLP-based embeddings
- Handle permutation-invariant data with GNNs

**Application**: Same inference task as 05a but with improved architecture that exploits spatial relationships.

## Data

Simulation data for Tutorials 05a and 05b are located in the `data/` directory:
- Generated using the [StreamSculptor](https://arxiv.org/abs/2410.21174v1) package
- Contains 10,000 simulated stellar streams with subhalo impacts
- Each stream has ~1,000-10,000 particles with 6D phase-space coordinates

## Credits

Tutorials created by Tri Nguyen with assistance from Claude Code.

## References

- **NPE/NLE/NRE**: [Cranmer et al. (2020) - "The frontier of simulation-based inference"](https://arxiv.org/abs/1911.01429)
- **TARP**: [Lemos et al. (2023) - "Sampling-Based Accuracy Testing of Posterior Estimators"](https://arxiv.org/abs/2302.03026)
- **Zuko**: [Neural Spline Flows library](https://zuko.readthedocs.io/)
- **StreamSculptor**: [Nibauer et al. (2024)](https://arxiv.org/abs/2410.21174v1)

## Additional Resources

- [SBI Toolkit Documentation](https://sbi-dev.github.io/sbi/)
- [Normalizing Flows Tutorial](https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial11/NF_image_modeling.html)
- [PyTorch Geometric Documentation](https://pytorch-geometric.readthedocs.io/)
