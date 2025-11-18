# Installation Instructions

## Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

## Installation Steps

### 1. Create a virtual environment (recommended)

```bash
python -m venv sbi-env
source sbi-env/bin/activate  # On Windows: sbi-env\Scripts\activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. (Optional) Install PyTorch with GPU support

If you have a CUDA-compatible GPU and want to accelerate training, install PyTorch with CUDA support:

```bash
# For CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

Run `nvidia-smi` in your terminal to check your GPU and CUDA version.

If you do not have a compatible GPU, install the CPU version:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

Visit [PyTorch's official website](https://pytorch.org/get-started/locally/) for more options.

### 4. Install PyTorch Geometric (for Tutorial 05b)

PyTorch Geometric requires additional installation steps:

```bash
pip install torch-geometric
pip install pyg-lib torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-$(python -c "import torch; print(torch.__version__)")+cpu.html
```

For GPU support, replace `cpu` with your CUDA version (e.g., `cu118` or `cu121`). Visit [PyTorch Geometric Installation](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html) for more details.

### 5. Verify installation

```bash
python -c "import torch; import numpy; import zuko; print('All packages installed successfully!')"
```

## Tutorial-Specific Requirements

- **Tutorials 01-04**: Only core packages required
- **Tutorial 05a**: Core packages + corner, tarp
- **Tutorial 05b**: All packages including torch-geometric

## Troubleshooting

If you encounter issues:

1. Make sure you're using Python 3.8 or higher
2. Update pip: `pip install --upgrade pip`
3. Try installing packages one at a time to identify the problematic package
4. Check PyTorch compatibility with your system at https://pytorch.org

## Alternative: Using conda

If you prefer conda:

```bash
conda create -n sbi-env python=3.10
conda activate sbi-env
pip install -r requirements.txt
```
