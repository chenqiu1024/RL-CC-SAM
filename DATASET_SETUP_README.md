# nnU-Net Dataset Setup Guide

This guide helps you download and setup a dataset for nnU-Net baseline experiments.

## 🚀 Quick Start

### 1. Test Your Environment
```bash
# Activate conda environment and set nnU-Net paths
conda activate llms
source setup_nnunet_env.sh

# Test if everything is working
python test_nnunet_setup.py
```

### 2. Download and Setup Dataset
```bash
# Install additional dependencies
pip install -r download_requirements.txt

# Run the download and setup script
python download_and_setup_dataset.py
```

### 3. Start Training
```bash
# Quick 2D training (fast, good for testing)
nnUNetv2_train 999 2d 0

# Or 3D training (slower, usually better results)
nnUNetv2_train 999 3d_fullres 0
```

## 📁 What Each Script Does

### `download_and_setup_dataset.py`
**Main script that:**
- Sets up nnU-Net environment variables automatically
- Tries to download Medical Segmentation Decathlon Task02_Heart dataset
- Falls back to creating a synthetic dataset if download fails
- Converts dataset to nnU-Net format
- Runs preprocessing
- Verifies everything is ready for training

**Usage:**
```bash
python download_and_setup_dataset.py
```

### `test_nnunet_setup.py`
**Quick verification script that:**
- Checks if environment variables are set
- Tests if nnU-Net commands are available
- Provides troubleshooting hints

**Usage:**
```bash
python test_nnunet_setup.py
```

### `setup_nnunet_env.sh`
**Environment setup script that:**
- Sets nnU-Net environment variables
- Creates required directories
- Shows current configuration

**Usage:**
```bash
source setup_nnunet_env.sh
```

## 🎯 Dataset Options

The script will try to download datasets in this order:

1. **Medical Segmentation Decathlon Task02_Heart**
   - Small real medical dataset (434.6 MB)
   - Good for baseline experiments
   - Downloaded from: http://medicaldecathlon.com/

2. **Synthetic Dataset (Fallback)**
   - Created automatically if download fails
   - 10 synthetic 3D cases (64x64x32 voxels)
   - Good for testing nnU-Net pipeline
   - Very fast to process

## 📋 Manual Dataset Download

If automatic download fails, you can manually download datasets:

### Option 1: Medical Segmentation Decathlon
1. Visit: http://medicaldecathlon.com/
2. Download any Task (e.g., Task02_Heart.tar)
3. Place in a temporary folder
4. Run: `nnUNetv2_convert_MSD_dataset -i /path/to/extracted/Task02_Heart`

### Option 2: Other Medical Datasets
- **BraTS Challenge:** https://www.synapse.org/#!Synapse:syn25829067
- **AMOS 2022:** https://amos22.grand-challenge.org/
- **KiTS Challenge:** https://kits-challenge.org/

## 🔧 Troubleshooting

### Environment Variables Not Set
```bash
# Make sure you sourced the script
source setup_nnunet_env.sh

# Check if variables are set
echo $nnUNet_raw
echo $nnUNet_preprocessed
echo $nnUNet_results
```

### nnU-Net Commands Not Available
```bash
# Make sure nnU-Net is installed
conda activate llms
pip install nnunetv2

# Test installation
nnUNetv2_plan_and_preprocess -h
```

### Download Issues
- Check internet connection
- Try manual download from http://medicaldecathlon.com/
- Script will create synthetic data as fallback

### Preprocessing Errors
- Make sure you have enough disk space (several GB)
- Check that dataset format is correct
- Look at error messages for specific issues

## 📊 Expected Results

After successful setup, you should have:

```
datasets/
├── nnUNet_raw/
│   └── Dataset999_Synthetic/  (or Dataset002_Heart)
│       ├── dataset.json
│       ├── imagesTr/
│       └── labelsTr/
├── nnUNet_preprocessed/
│   └── Dataset999_Synthetic/
│       ├── nnUNetPlans.json
│       └── dataset_fingerprint.json
└── nnUNet_results/
    └── (will be populated during training)
```

## 🎯 Training Commands

Once setup is complete:

```bash
# Quick test (2D, single fold)
nnUNetv2_train 999 2d 0

# Full 3D training
nnUNetv2_train 999 3d_fullres 0

# Cross-validation (all 5 folds)
for fold in 0 1 2 3 4; do
    nnUNetv2_train 999 2d $fold
done

# Find best configuration
nnUNetv2_find_best_configuration 999

# Run inference
nnUNetv2_predict -i INPUT_FOLDER -o OUTPUT_FOLDER -d 999 -c 2d -f 0
```

## 💡 Tips for Baseline Experiments

1. **Start with 2D training** - faster, good for initial testing
2. **Use synthetic data first** - ensures pipeline works before using large datasets
3. **Monitor training progress** - check logs for convergence
4. **Try different configurations** - 2d vs 3d_fullres
5. **Validate results** - use cross-validation for robust evaluation

## 📚 Additional Resources

- [nnU-Net Documentation](https://github.com/MIC-DKFZ/nnUNet)
- [nnU-Net Paper](https://www.nature.com/articles/s41592-020-01008-z)
- [Medical Segmentation Decathlon](http://medicaldecathlon.com/)
- [nnU-Net Tutorial](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/how_to_use_nnunet.md) 