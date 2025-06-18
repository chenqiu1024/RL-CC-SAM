#!/bin/bash

# nnU-Net Environment Variables Setup Script
# This script sets up the required environment variables for nnU-Net
# 
# Usage:
#   source setup_nnunet_env.sh
#   OR
#   . setup_nnunet_env.sh
#
# Note: Use 'source' or '.' to run this script so that the environment 
# variables are set in your current shell session.

# Get the absolute path of the current directory (where this script is located)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Set nnU-Net environment variables
export nnUNet_raw="${SCRIPT_DIR}/datasets/nnUNet_raw"
export nnUNet_preprocessed="${SCRIPT_DIR}/datasets/nnUNet_preprocessed"
export nnUNet_results="${SCRIPT_DIR}/datasets/nnUNet_results"

# Create directories if they don't exist
mkdir -p "$nnUNet_raw"
mkdir -p "$nnUNet_preprocessed"
mkdir -p "$nnUNet_results"

# Display the set environment variables
echo "✅ nnU-Net environment variables have been set:"
echo "   nnUNet_raw: $nnUNet_raw"
echo "   nnUNet_preprocessed: $nnUNet_preprocessed"
echo "   nnUNet_results: $nnUNet_results"
echo ""
echo "📁 Directory structure:"
echo "   $(ls -la datasets/)"
echo ""
echo "🔍 To verify the variables are set, you can run:"
echo "   echo \$nnUNet_raw"
echo "   echo \$nnUNet_preprocessed"
echo "   echo \$nnUNet_results"
echo ""
echo "💡 To make these permanent, add the export commands to your ~/.zshrc file"
echo "   (since you're using zsh shell)" 