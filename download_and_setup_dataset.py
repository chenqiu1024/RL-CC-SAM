#!/usr/bin/env python3
"""
Medical Segmentation Decathlon Dataset Download and Setup Script for nnU-Net
This script downloads Task02_Heart (smallest MSD dataset) and prepares it for nnU-Net training.

Usage:
    python download_and_setup_dataset.py
    
Requirements:
    - nnU-Net installed in environment
    - Environment variables set (run: source setup_nnunet_env.sh)
"""

import os
import sys
import subprocess
import tarfile
import shutil
import requests
from pathlib import Path
from urllib.parse import urlparse
import tempfile
import time
from tqdm import tqdm

def setup_environment():
    """Set up nnU-Net environment variables if not already set."""
    current_dir = Path(__file__).parent.absolute()
    
    env_vars = {
        'nnUNet_raw': str(current_dir / 'datasets' / 'nnUNet_raw'),
        'nnUNet_preprocessed': str(current_dir / 'datasets' / 'nnUNet_preprocessed'),
        'nnUNet_results': str(current_dir / 'datasets' / 'nnUNet_results')
    }
    
    for key, value in env_vars.items():
        if key not in os.environ:
            os.environ[key] = value
            print(f"✅ Set {key} = {value}")
        else:
            print(f"🔄 Using existing {key} = {os.environ[key]}")
    
    # Create directories
    for path in env_vars.values():
        Path(path).mkdir(parents=True, exist_ok=True)
    
    return env_vars

def download_file_with_progress(url, destination):
    """Download a file with progress bar."""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(destination, 'wb') as file, tqdm(
        desc=Path(destination).name,
        total=total_size,
        unit='B',
        unit_scale=True,
        unit_divisor=1024,
    ) as pbar:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                file.write(chunk)
                pbar.update(len(chunk))

def download_msd_heart_dataset(download_dir):
    """Download the MSD Task02_Heart dataset."""
    print("📥 Downloading Medical Segmentation Decathlon - Task02_Heart...")
    
    # Direct download URL for Task02_Heart (you may need to update this)
    # Alternative sources for MSD datasets
    urls = [
        "http://medicaldecathlon.com/files/Task02_Heart.tar",  # Primary source
        # Add backup URLs if available
    ]
    
    tar_file = download_dir / "Task02_Heart.tar"
    
    # Try to download from available sources
    downloaded = False
    for url in urls:
        try:
            print(f"🌐 Trying to download from: {url}")
            download_file_with_progress(url, tar_file)
            downloaded = True
            break
        except Exception as e:
            print(f"❌ Failed to download from {url}: {e}")
            continue
    
    if not downloaded:
        print("❌ Could not download from any source.")
        print("📋 Manual download instructions:")
        print("1. Visit: http://medicaldecathlon.com/")
        print("2. Download Task02_Heart.tar")
        print(f"3. Place it in: {tar_file}")
        print("4. Run this script again")
        
        # Check if user has manually downloaded the file
        if not tar_file.exists():
            user_input = input("Have you manually downloaded the file? (y/n): ")
            if user_input.lower() != 'y':
                sys.exit(1)
    
    return tar_file

def download_alternative_dataset(download_dir):
    """Download an alternative dataset if MSD is not available."""
    print("📥 Downloading alternative dataset for nnU-Net testing...")
    
    # Use a small publicly available medical dataset
    # Example: CHAOS Challenge data or create synthetic data
    
    # Create a minimal synthetic dataset for testing
    create_synthetic_dataset(download_dir)
    
    return download_dir / "synthetic_dataset"

def create_synthetic_dataset(download_dir):
    """Create a small synthetic dataset for testing nnU-Net pipeline."""
    import numpy as np
    import nibabel as nib
    import json
    
    print("🔬 Creating synthetic dataset for nnU-Net testing...")
    
    dataset_dir = download_dir / "Dataset999_Synthetic"
    images_tr = dataset_dir / "imagesTr"
    labels_tr = dataset_dir / "labelsTr"
    
    # Create directories
    images_tr.mkdir(parents=True, exist_ok=True)
    labels_tr.mkdir(parents=True, exist_ok=True)
    
    # Create synthetic images and labels
    for i in range(10):  # 10 cases for quick testing
        case_id = f"case_{i:03d}"
        
        # Create 3D image (64x64x32) - small for quick processing
        image_data = np.random.randint(0, 1000, (64, 64, 32), dtype=np.int16)
        # Add some structure
        image_data[20:40, 20:40, 10:20] += 200  # "organ"
        
        # Create corresponding segmentation
        seg_data = np.zeros((64, 64, 32), dtype=np.uint8)
        seg_data[25:35, 25:35, 12:18] = 1  # "organ" label
        
        # Save as NIfTI files
        img_nifti = nib.Nifti1Image(image_data, affine=np.eye(4))
        seg_nifti = nib.Nifti1Image(seg_data, affine=np.eye(4))
        
        nib.save(img_nifti, images_tr / f"{case_id}_0000.nii.gz")
        nib.save(seg_nifti, labels_tr / f"{case_id}.nii.gz")
    
    # Create dataset.json
    dataset_json = {
        "channel_names": {"0": "synthetic"},
        "labels": {"background": 0, "organ": 1},
        "numTraining": 10,
        "file_ending": ".nii.gz"
    }
    
    with open(dataset_dir / "dataset.json", 'w') as f:
        json.dump(dataset_json, f, indent=2)
    
    print(f"✅ Created synthetic dataset at: {dataset_dir}")
    return dataset_dir

def extract_dataset(tar_file, extract_to):
    """Extract tar file to destination."""
    print(f"📂 Extracting {tar_file.name}...")
    
    with tarfile.open(tar_file, 'r') as tar:
        tar.extractall(extract_to)
    
    print(f"✅ Extracted to: {extract_to}")
    
    # Find the extracted folder
    extracted_folders = [d for d in extract_to.iterdir() if d.is_dir()]
    if extracted_folders:
        return extracted_folders[0]
    return extract_to

def convert_to_nnunet_format(dataset_path, target_id=999):
    """Convert dataset to nnU-Net format."""
    print(f"🔄 Converting dataset to nnU-Net format...")
    
    raw_data_folder = Path(os.environ['nnUNet_raw'])
    
    if 'Task02_Heart' in str(dataset_path):
        # Use MSD converter for official MSD data
        cmd = [
            'nnUNetv2_convert_MSD_dataset',
            '-i', str(dataset_path),
            '-overwrite_id', str(target_id)
        ]
    else:
        # For synthetic or other datasets, copy directly
        target_name = f"Dataset{target_id:03d}_Synthetic"
        target_path = raw_data_folder / target_name
        
        if target_path.exists():
            shutil.rmtree(target_path)
        
        shutil.copytree(dataset_path, target_path)
        print(f"✅ Copied dataset to: {target_path}")
        return target_id
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("✅ Successfully converted to nnU-Net format")
        return target_id
    except subprocess.CalledProcessError as e:
        print(f"❌ Conversion failed: {e}")
        print(f"Error output: {e.stderr}")
        return None

def run_preprocessing(dataset_id):
    """Run nnU-Net preprocessing."""
    print(f"⚙️ Running nnU-Net preprocessing for dataset {dataset_id}...")
    
    cmd = [
        'nnUNetv2_plan_and_preprocess',
        '-d', str(dataset_id),
        '--verify_dataset_integrity'
    ]
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("✅ Preprocessing completed successfully")
        print("🔍 Dataset integrity verified")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Preprocessing failed: {e}")
        print(f"Error output: {e.stderr}")
        return False

def verify_setup(dataset_id):
    """Verify that everything is set up correctly."""
    print("🔍 Verifying setup...")
    
    # Check environment variables
    required_vars = ['nnUNet_raw', 'nnUNet_preprocessed', 'nnUNet_results']
    for var in required_vars:
        if var in os.environ:
            print(f"✅ {var}: {os.environ[var]}")
        else:
            print(f"❌ {var}: Not set")
            return False
    
    # Check if dataset exists
    raw_path = Path(os.environ['nnUNet_raw']) / f"Dataset{dataset_id:03d}_*"
    preprocessed_path = Path(os.environ['nnUNet_preprocessed']) / f"Dataset{dataset_id:03d}_*"
    
    import glob
    raw_datasets = glob.glob(str(raw_path))
    preprocessed_datasets = glob.glob(str(preprocessed_path))
    
    if raw_datasets:
        print(f"✅ Raw dataset found: {raw_datasets[0]}")
    else:
        print(f"❌ Raw dataset not found at: {raw_path}")
        return False
    
    if preprocessed_datasets:
        print(f"✅ Preprocessed dataset found: {preprocessed_datasets[0]}")
    else:
        print(f"❌ Preprocessed dataset not found at: {preprocessed_path}")
        return False
    
    return True

def print_training_instructions(dataset_id):
    """Print instructions for training."""
    print("\n" + "="*60)
    print("🎯 TRAINING INSTRUCTIONS")
    print("="*60)
    print(f"Your dataset (ID: {dataset_id}) is ready for training!")
    print("\n📋 To train a model, run:")
    print(f"   nnUNetv2_train {dataset_id} 2d 0")
    print(f"   nnUNetv2_train {dataset_id} 3d_fullres 0")
    
    print("\n🔍 To find the best configuration:")
    print(f"   nnUNetv2_find_best_configuration {dataset_id}")
    
    print("\n📊 To run inference:")
    print(f"   nnUNetv2_predict -i INPUT_FOLDER -o OUTPUT_FOLDER -d {dataset_id} -c CONFIG -f FOLD")
    
    print("\n📚 Useful commands:")
    print("   nnUNetv2_train -h                    # Help for training")
    print("   nnUNetv2_predict -h                 # Help for prediction")
    print(f"   ls $nnUNet_preprocessed/Dataset{dataset_id:03d}_*/     # Check preprocessed data")
    
    print("\n💡 Tips:")
    print("   - Start with 2d training for faster testing")
    print("   - 3d_fullres usually gives better results")
    print("   - Use fold 0 for quick testing, folds 0-4 for full cross-validation")

def main():
    """Main function to download and setup dataset."""
    print("🚀 nnU-Net Dataset Download and Setup Script")
    print("=" * 50)
    
    # Step 1: Setup environment
    print("\n1️⃣ Setting up environment...")
    env_vars = setup_environment()
    
    # Step 2: Create temporary download directory
    download_dir = Path(tempfile.mkdtemp(prefix="nnunet_download_"))
    print(f"📁 Using temporary directory: {download_dir}")
    
    try:
        # Step 3: Download dataset
        print("\n2️⃣ Downloading dataset...")
        
        # Try to download MSD Heart dataset first
        dataset_path = None
        dataset_id = 2  # Default for Heart dataset
        
        try:
            tar_file = download_msd_heart_dataset(download_dir)
            if tar_file.exists():
                dataset_path = extract_dataset(tar_file, download_dir)
        except Exception as e:
            print(f"⚠️ MSD download failed: {e}")
        
        # If MSD download failed, create synthetic dataset
        if not dataset_path or not dataset_path.exists():
            print("📊 Creating synthetic dataset for testing...")
            dataset_path = create_synthetic_dataset(download_dir)
            dataset_id = 999  # Use ID 999 for synthetic data
        
        # Step 4: Convert to nnU-Net format
        print("\n3️⃣ Converting to nnU-Net format...")
        converted_id = convert_to_nnunet_format(dataset_path, dataset_id)
        
        if not converted_id:
            print("❌ Dataset conversion failed")
            return False
        
        # Step 5: Run preprocessing
        print("\n4️⃣ Running preprocessing...")
        if not run_preprocessing(converted_id):
            print("❌ Preprocessing failed")
            return False
        
        # Step 6: Verify setup
        print("\n5️⃣ Verifying setup...")
        if verify_setup(converted_id):
            print("✅ Setup verification successful!")
            print_training_instructions(converted_id)
            return True
        else:
            print("❌ Setup verification failed")
            return False
    
    finally:
        # Cleanup temporary directory
        print(f"\n🧹 Cleaning up temporary directory: {download_dir}")
        shutil.rmtree(download_dir, ignore_errors=True)

if __name__ == "__main__":
    try:
        success = main()
        if success:
            print("\n🎉 Dataset setup completed successfully!")
            print("You can now start training your nnU-Net models.")
        else:
            print("\n❌ Dataset setup failed. Check the error messages above.")
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n⏹️ Setup interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        sys.exit(1) 