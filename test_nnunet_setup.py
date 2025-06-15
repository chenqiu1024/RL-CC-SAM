#!/usr/bin/env python3
"""
Quick test script to verify nnU-Net environment setup
"""

import os
import subprocess
import sys

def test_environment():
    """Test if nnU-Net environment is properly set up."""
    print("🔍 Testing nnU-Net Environment Setup")
    print("=" * 40)
    
    # Check environment variables
    print("\n📍 Environment Variables:")
    required_vars = ['nnUNet_raw', 'nnUNet_preprocessed', 'nnUNet_results']
    all_vars_ok = True
    
    for var in required_vars:
        if var in os.environ:
            print(f"   ✅ {var}: {os.environ[var]}")
        else:
            print(f"   ❌ {var}: Not set")
            all_vars_ok = False
    
    if not all_vars_ok:
        print("\n💡 To set environment variables, run: source setup_nnunet_env.sh")
        return False
    
    # Test nnU-Net commands
    print("\n🔧 Testing nnU-Net Commands:")
    try:
        result = subprocess.run(['nnUNetv2_plan_and_preprocess', '-h'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print("   ✅ nnUNetv2_plan_and_preprocess: Available")
        else:
            print("   ❌ nnUNetv2_plan_and_preprocess: Error")
            return False
    except Exception as e:
        print(f"   ❌ nnUNet commands not available: {e}")
        print("   💡 Make sure nnU-Net is installed: pip install nnunetv2")
        return False
    
    print("\n🎉 Environment setup is correct!")
    print("\n📋 Next steps:")
    print("   1. Run: python download_and_setup_dataset.py")
    print("   2. Or use the Jupyter notebook: notebooks/nnunet_dataset_setup.ipynb")
    
    return True

if __name__ == "__main__":
    success = test_environment()
    sys.exit(0 if success else 1) 