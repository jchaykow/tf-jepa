#!/usr/bin/env python3
"""
CLI tool for easily finetuning NormWear on available datasets.

Usage:
    python finetune_cli.py --dataset Epilepsy --epochs 50
    python finetune_cli.py --dataset EMG --batch_size 16
    python finetune_cli.py --list-datasets
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

# Check for required dependencies
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch not installed. Some features may be limited.")

try:
    from downstream_pipeline.config import DATASET_CONFIG
except ImportError:
    print("Warning: Cannot import DATASET_CONFIG. Using fallback configuration.")
    # Fallback configuration
    DATASET_CONFIG = {
        "Epilepsy": {"n_ch": 1, "n_cl": 2, "task": "class", "lr": 1e-2, "bs": 16, "max_len": 178},
        "EMG": {"n_ch": 1, "n_cl": 3, "task": "class", "lr": 1e-3, "bs": 32, "max_len": 96},
        "FD": {"n_ch": 1, "n_cl": 3, "task": "class", "lr": 1e-3, "bs": 8, "max_len": 21},
        "Gesture": {"n_ch": 3, "n_cl": 8, "task": "class", "lr": 1e-3, "bs": 32, "max_len": 315},
        "SleepEEG": {"n_ch": 1, "n_cl": 1, "task": "class", "lr": 1e-3, "bs": 64, "max_len": 200}
    }

# Model weights URL
MODEL_WEIGHTS_URL = "https://github.com/Mobile-Sensing-and-UbiComp-Laboratory/NormWear/releases/download/v1.0.0-alpha/normwear_last_checkpoint-15470-correct.pth"
MODEL_WEIGHTS_PATH = "weights/normwear_pretrained.pth"

# Available datasets
AVAILABLE_DATASETS = {
    "Epilepsy": {
        "download_url": "https://figshare.com/ndownloader/articles/19930199/versions/2",
        "convert_script": "data_converters/convert_epilespy_data.py",
        "data_dir": "Epilepsy"
    },
    "EMG": {
        "download_url": "https://figshare.com/ndownloader/articles/19930250/versions/1",
        "convert_script": "data_converters/convert_emg_data.py",
        "data_dir": "EMG"
    },
    "FD": {
        "download_url": "https://figshare.com/ndownloader/articles/19930226/versions/1",
        "convert_script": "data_converters/convert_fd_data.py",
        "data_dir": "FD"
    },
    "Gesture": {
        "download_url": "https://figshare.com/ndownloader/articles/19930247/versions/1",
        "convert_script": "data_converters/convert_gesture_data.py",
        "data_dir": "Gesture"
    },
    "SleepEEG": {
        "download_url": "https://figshare.com/ndownloader/articles/19930178/versions/1",
        "convert_script": "data_converters/convert_sleepeeg_data.py",
        "data_dir": "SleepEEG"
    }
}

def run_command(cmd, description="Running command"):
    """Run a shell command and handle errors."""
    print(f"\n{description}...")
    print(f"Command: {cmd}")
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        if result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
        if e.stderr:
            print(f"Error output: {e.stderr}")
        return False

def download_model_weights():
    """Download the pretrained NormWear model weights."""
    weights_dir = Path("weights")
    weights_dir.mkdir(exist_ok=True)
    
    if Path(MODEL_WEIGHTS_PATH).exists():
        print(f"Model weights already exist at {MODEL_WEIGHTS_PATH}")
        return True
    
    print(f"Downloading model weights from {MODEL_WEIGHTS_URL}")
    cmd = f"wget -O {MODEL_WEIGHTS_PATH} '{MODEL_WEIGHTS_URL}'"
    return run_command(cmd, "Downloading model weights")

def download_dataset(dataset_name):
    """Download a specific dataset."""
    if dataset_name not in AVAILABLE_DATASETS:
        print(f"Error: Dataset '{dataset_name}' not found.")
        print(f"Available datasets: {', '.join(AVAILABLE_DATASETS.keys())}")
        return False
    
    dataset_info = AVAILABLE_DATASETS[dataset_name]
    data_dir = Path(f"data/{dataset_info['data_dir']}")
    
    # Check if data already exists
    if data_dir.exists() and any(data_dir.iterdir()):
        print(f"Dataset {dataset_name} already exists in {data_dir}")
        return True
    
    # Create data directory
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Download dataset
    zip_file = f"{dataset_name}.zip"
    download_cmd = f'wget --user-agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36" -O {zip_file} "{dataset_info["download_url"]}"'
    
    if not run_command(download_cmd, f"Downloading {dataset_name} dataset"):
        return False
    
    # Extract dataset
    extract_cmd = f"unzip {zip_file} -d {data_dir}"
    if not run_command(extract_cmd, f"Extracting {dataset_name} dataset"):
        return False
    
    # Clean up zip file
    os.remove(zip_file)
    
    return True

def prepare_dataset(dataset_name):
    """Convert dataset to NormWear format."""
    if dataset_name not in AVAILABLE_DATASETS:
        return False
    
    dataset_info = AVAILABLE_DATASETS[dataset_name]
    convert_script = dataset_info["convert_script"]
    
    # Check if conversion script exists
    if not Path(convert_script).exists():
        print(f"Error: Conversion script {convert_script} not found.")
        return False
    
    # Check if data is already prepared
    prepared_data_path = Path(f"data/{dataset_name.lower()}/sample_for_downstream")
    if prepared_data_path.exists() and any(prepared_data_path.iterdir()):
        print(f"Dataset {dataset_name} is already prepared.")
        return True
    
    # Run conversion script
    cmd = f"uv run python {convert_script}"
    return run_command(cmd, f"Preparing {dataset_name} dataset for NormWear")

def run_finetuning(args):
    """Run the finetuning process."""
    dataset_name = args.dataset
    
    # Verify dataset exists in config
    if dataset_name not in DATASET_CONFIG:
        print(f"Error: Dataset '{dataset_name}' not found in DATASET_CONFIG.")
        print(f"Available datasets in config: {', '.join(DATASET_CONFIG.keys())}")
        return False
    
    # Build finetuning command
    cmd_parts = [
        "uv run python -m downstream_pipeline.main_finetune",
        f"--ds_name {dataset_name}",
        f"--data_dir data",
        f"--checkpoint {MODEL_WEIGHTS_PATH}",
        f"--epochs {args.epochs}",
        f"--remark {args.remark}"
    ]
    
    # Add optional arguments
    if args.batch_size:
        cmd_parts.append(f"--batch_size {args.batch_size}")
    if args.learning_rate:
        cmd_parts.append(f"--lr {args.learning_rate}")
    if args.gpu:
        cmd_parts.append(f"--device cuda:{args.gpu}")
    else:
        if TORCH_AVAILABLE:
            cmd_parts.append("--device cuda" if torch.cuda.is_available() else "--device cpu")
        else:
            cmd_parts.append("--device cpu")
    
    # Add multi-GPU support
    if args.num_gpus > 1:
        # For multi-GPU, we need to use torchrun through uv
        # Extract the module and arguments from the command
        module_and_args = " ".join(cmd_parts).replace("uv run python -m ", "")
        cmd = f"uv run torchrun --nproc_per_node={args.num_gpus} --module {module_and_args}"
    else:
        cmd = " ".join(cmd_parts)
    
    return run_command(cmd, f"Finetuning NormWear on {dataset_name}")

def list_datasets():
    """List all available datasets."""
    print("\nAvailable datasets for finetuning:")
    print("-" * 50)
    for name, info in AVAILABLE_DATASETS.items():
        config = DATASET_CONFIG.get(name, {})
        print(f"\n{name}:")
        print(f"  - Channels: {config.get('n_ch', 'N/A')}")
        print(f"  - Classes: {config.get('n_cl', 'N/A')}")
        print(f"  - Task: {config.get('task', 'N/A')}")
        print(f"  - Default batch size: {config.get('bs', 'N/A')}")
        print(f"  - Max sequence length: {config.get('max_len', 'N/A')}")

def main():
    parser = argparse.ArgumentParser(
        description="CLI tool for finetuning NormWear on various datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List available datasets
  uv run python finetune_cli.py --list-datasets
  
  # Finetune on Epilepsy dataset with default settings
  uv run python finetune_cli.py --dataset Epilepsy
  
  # Finetune with custom settings
  uv run python finetune_cli.py --dataset EMG --epochs 100 --batch-size 32 --lr 0.001
  
  # Use multiple GPUs
  uv run python finetune_cli.py --dataset Gesture --num-gpus 4
  
  # Skip download/preparation if data exists
  uv run python finetune_cli.py --dataset FD --skip-download --skip-prepare
        """
    )
    
    parser.add_argument("--dataset", type=str, choices=list(AVAILABLE_DATASETS.keys()),
                        help="Dataset to finetune on")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of training epochs (default: 50)")
    parser.add_argument("--batch-size", type=int, dest="batch_size",
                        help="Batch size (uses dataset default if not specified)")
    parser.add_argument("--lr", "--learning-rate", type=float, dest="learning_rate",
                        help="Learning rate (uses dataset default if not specified)")
    parser.add_argument("--remark", type=str, default="finetune",
                        help="Remark for experiment tracking (default: finetune)")
    parser.add_argument("--gpu", type=int,
                        help="GPU device ID to use (default: auto-detect)")
    parser.add_argument("--num-gpus", type=int, default=1,
                        help="Number of GPUs for distributed training (default: 1)")
    parser.add_argument("--skip-download", action="store_true",
                        help="Skip downloading model weights and dataset")
    parser.add_argument("--skip-prepare", action="store_true",
                        help="Skip dataset preparation")
    parser.add_argument("--list-datasets", action="store_true",
                        help="List all available datasets")
    
    args = parser.parse_args()
    
    # Handle list datasets
    if args.list_datasets:
        list_datasets()
        return
    
    # Require dataset for finetuning
    if not args.dataset:
        parser.error("--dataset is required for finetuning. Use --list-datasets to see available options.")
    
    print(f"\n🚀 NormWear Finetuning CLI")
    print(f"=" * 50)
    print(f"Dataset: {args.dataset}")
    print(f"Epochs: {args.epochs}")
    if TORCH_AVAILABLE:
        print(f"Device: {'GPU' if torch.cuda.is_available() else 'CPU'}")
    else:
        print(f"Device: CPU (PyTorch not installed)")
    print(f"=" * 50)
    
    # Step 1: Download model weights
    if not args.skip_download:
        if not download_model_weights():
            print("Failed to download model weights. Exiting.")
            sys.exit(1)
    
    # Step 2: Download dataset
    if not args.skip_download:
        if not download_dataset(args.dataset):
            print(f"Failed to download {args.dataset} dataset. Exiting.")
            sys.exit(1)
    
    # Step 3: Prepare dataset
    if not args.skip_prepare:
        if not prepare_dataset(args.dataset):
            print(f"Failed to prepare {args.dataset} dataset. Exiting.")
            sys.exit(1)
    
    # Step 4: Run finetuning
    print(f"\n🎯 Starting finetuning on {args.dataset}...")
    if run_finetuning(args):
        print(f"\n✅ Finetuning completed successfully!")
        print(f"Results saved in: data/results/Normwear_Downstream/{args.dataset}/")
    else:
        print(f"\n❌ Finetuning failed.")
        sys.exit(1)

if __name__ == "__main__":
    main()