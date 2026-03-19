#!/bin/bash
# Setup script for NormWear finetuning environment

echo "🔧 Setting up NormWear environment..."

# Check if uv is installed
if command -v uv &> /dev/null; then
    echo "✅ uv is installed"
    echo "Installing dependencies with uv..."
    uv sync
    
    echo "✅ Environment setup complete!"
    echo ""
    echo "To run the finetuning CLI:"
    echo "  uv run python finetune_cli.py --list-datasets"
    echo "  uv run python finetune_cli.py --dataset Epilepsy"
else
    echo "❌ uv not found. Please install uv first:"
    echo "  curl -LsSf https://astral.sh/uv/install.sh | sh"
    echo ""
    echo "Or install with pip:"
    echo "  pip install uv"
    exit 1
fi