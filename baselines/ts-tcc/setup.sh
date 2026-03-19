#!/bin/bash

# TS-TCC Quick Setup Script
# This script sets up the development environment for TS-TCC experiments

set -e  # Exit on error

echo "=========================================="
echo "TS-TCC Development Environment Setup"
echo "=========================================="

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "Installing uv package manager..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    echo "✓ uv installed successfully!"
    echo ""
    echo "Please restart your terminal or run: source ~/.bashrc (or ~/.zshrc)"
    echo "Then run this script again."
    exit 0
fi

echo "✓ uv is already installed"

# Install dependencies
echo ""
echo "Installing Python dependencies..."
uv add torch numpy scikit-learn pandas tqdm einops openpyxl

echo ""
echo "✓ Dependencies installed successfully!"

# Make scripts executable
echo ""
echo "Making scripts executable..."
chmod +x run_experiment.sh
chmod +x download_datasets.sh

echo "✓ Scripts are now executable"

# Check for datasets
echo ""
echo "Checking for datasets..."
if [ -f "src/ts_tcc/data/SleepEEG/train.pt" ] && [ -f "src/ts_tcc/data/Epilepsy/train.pt" ]; then
    echo "✓ Some datasets are already downloaded"
else
    echo ""
    echo "No datasets found. Would you like to download them now? (y/N)"
    read -r response
    if [[ "$response" =~ ^[Yy]$ ]]; then
        echo "Downloading datasets..."
        bash download_datasets.sh
    else
        echo "You can download datasets later with: bash download_datasets.sh"
    fi
fi

echo ""
echo "=========================================="
echo "✓ Setup completed successfully!"
echo "=========================================="
echo ""
echo "You can now run experiments using:"
echo "  uv run python run_experiment.py sleepeeg_to_epilepsy"
echo "  ./run_experiment.sh sleepeeg_to_epilepsy"
echo ""
echo "For more information, see CLI_USAGE.md"