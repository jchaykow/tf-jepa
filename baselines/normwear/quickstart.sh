#!/bin/bash
# Quick start script for NormWear finetuning
# This script sets up everything and runs a quick test

echo "🚀 NormWear Quickstart - Complete Setup and Test Run"
echo "===================================================="

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is required but not installed. Please install Python 3.12+"
    exit 1
fi

# Step 1: Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "❌ uv is required but not installed."
    echo "Please install uv first:"
    echo "  curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

# Step 2: Setup environment if not already done
if [ ! -d ".venv" ]; then
    echo "📦 Setting up Python environment with uv..."
    uv sync
    if [ $? -ne 0 ]; then
        echo "❌ Failed to setup environment"
        exit 1
    fi
else
    echo "✅ Environment already set up"
fi

# Step 3: Quick test - list datasets
echo ""
echo "📋 Available datasets:"
uv run python finetune_cli.py --list-datasets

# Step 3: Run a quick 2-epoch test on Epilepsy dataset
echo ""
echo "🧪 Running quick test (2 epochs on Epilepsy dataset)..."
echo "This will:"
echo "  1. Download model weights (~500MB) if needed"
echo "  2. Download Epilepsy dataset if needed"
echo "  3. Prepare the data"
echo "  4. Run 2 epochs of finetuning (~25 minutes on CPU)"
echo ""
read -p "Continue? (y/n) " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    uv run python finetune_cli.py --dataset Epilepsy --epochs 2 --remark quickstart_test
    
    if [ $? -eq 0 ]; then
        echo ""
        echo "✅ Quick test completed successfully!"
        echo ""
        echo "📊 Results saved in: data/results/Normwear_Downstream/Epilepsy/"
        echo ""
        echo "🎯 Next steps:"
        echo "  - Run full training: uv run python finetune_cli.py --dataset Epilepsy --epochs 50"
        echo "  - Try other datasets: uv run python finetune_cli.py --dataset EMG"
        echo "  - See all options: uv run python finetune_cli.py --help"
    else
        echo "❌ Test failed. Please check the error messages above."
    fi
else
    echo "Test cancelled."
    echo ""
    echo "You can run finetuning manually with:"
    echo "  uv run python finetune_cli.py --dataset Epilepsy"
fi