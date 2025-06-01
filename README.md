# RL-CC-SAM

This project combines Reinforcement Learning with Segment Anything Model (SAM) for interactive image segmentation.

## Setup

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/RL-CC-SAM.git
cd RL-CC-SAM
```

### 2. Initialize Submodules

The project uses Facebook's Segment Anything Model (SAM) as a submodule. Initialize it with:

```bash
git submodule update --init --recursive
```

### 3. Download Model Checkpoint

Download the SAM model checkpoint and place it in the `pretrained` directory:

```bash
mkdir -p pretrained
# Download the model checkpoint (choose one):
# - ViT-H SAM model (2.4GB): https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
# - ViT-L SAM model (1.2GB): https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth
# - ViT-B SAM model (375MB): https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth

# Example for downloading ViT-H model:
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth -O pretrained/sam_vit_h_4b8939.pth
```

### 4. Install Dependencies

#### Option 1: Using venv (Recommended for most users)

```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Unix/macOS
# or
.\venv\Scripts\activate   # On Windows

# Install dependencies
pip install -r requirements.txt
```

#### Option 2: Using Anaconda

```bash
# Create and activate conda environment
conda create -n rl-cc-sam python=3.10
conda activate rl-cc-sam

# Install PyTorch with CUDA support (if you have NVIDIA GPU)
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
# Or for CPU-only
# conda install pytorch torchvision torchaudio cpuonly -c pytorch
# Or for MacOS, refer to https://pytorch.org/get-started/locally/ for PyTorch installation

# Install other dependencies
pip install -r requirements.txt
```

#### Option 3: Using Mamba (Faster alternative to conda)

```bash
# Create and activate mamba environment
mamba create -n rl-cc-sam python=3.10
mamba activate rl-cc-sam

# Install PyTorch with CUDA support (if you have NVIDIA GPU)
mamba install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
# Or for CPU-only
# mamba install pytorch torchvision torchaudio cpuonly -c pytorch

# Install other dependencies
pip install -r requirements.txt
```

#### Environment Management Tips

- To deactivate the environment:
  ```bash
  # For venv
  deactivate
  
  # For conda/mamba
  conda deactivate
  # or
  mamba deactivate
  ```

- To remove the environment:
  ```bash
  # For venv
  rm -rf venv/  # On Unix/macOS
  # or
  rmdir /s /q venv  # On Windows
  
  # For conda/mamba
  conda env remove -n rl-cc-sam
  # or
  mamba env remove -n rl-cc-sam
  ```

- To update dependencies:
  ```bash
  pip install --upgrade -r requirements.txt
  ```

## Project Structure

```
RL-CC-SAM/
├── pretrained/           # Model checkpoints (not tracked by git)
├── segment-anything/     # SAM submodule
├── src/                  # Source code
├── tests/               # Test files
├── notebooks/           # Jupyter notebooks
├── requirements.txt     # Python dependencies
└── README.md           # This file
```

## Usage

[Add usage instructions here]

## Development

[Add development guidelines here]

## License

[Add license information here]

## Acknowledgments

- [Segment Anything Model (SAM)](https://github.com/facebookresearch/segment-anything) by Meta AI Research
- [Add other acknowledgments here] 