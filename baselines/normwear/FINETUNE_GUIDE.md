# NormWear Finetuning CLI Guide

This guide explains how to use the `finetune_cli.py` script to easily finetune the NormWear model on various datasets.

## Prerequisites

- Python 3.12+ (check with `python --version`)
- uv (Python package manager) - [Install instructions](https://github.com/astral-sh/uv)
- Git (for cloning the repository)
- ~10GB free disk space (for model weights and datasets)
- (Optional) CUDA-capable GPU for faster training

## Complete Setup for New Developers

### 1. Set Up Environment
```bash
# Install dependencies using uv (recommended)
./setup_environment.sh

# Or manually with uv
uv sync
```

### 2. Quick Start - One Command Finetuning
```bash
# This single command will:
# 1. Download the pretrained model weights (~500MB)
# 2. Download the Epilepsy dataset
# 3. Prepare the data in NormWear format
# 4. Run finetuning for 50 epochs
uv run python finetune_cli.py --dataset Epilepsy
```

## Quick Examples

1. **List available datasets:**
   ```bash
   uv run python finetune_cli.py --list-datasets
   ```

2. **Finetune with custom settings:**
   ```bash
   uv run python finetune_cli.py --dataset EMG --epochs 100 --batch-size 32 --lr 0.001
   ```

3. **Quick test run (2 epochs):**
   ```bash
   uv run python finetune_cli.py --dataset Epilepsy --epochs 2 --remark quick_test
   ```

## What the Script Does

The CLI script automates the entire finetuning workflow:

1. **Downloads model weights** from the official NormWear release
2. **Downloads the dataset** from the appropriate source
3. **Prepares the dataset** by converting it to NormWear format
4. **Runs finetuning** with configurable parameters

## Available Datasets

- **Epilepsy**: EEG data for epilepsy classification (1 channel, 2 classes)
- **EMG**: Electromyography data (1 channel, 3 classes)
- **FD**: Fault Detection data (1 channel, 3 classes)
- **Gesture**: Motion sensor data for gesture recognition (3 channels, 8 classes)
- **SleepEEG**: Sleep stage classification from EEG (1 channel)

## Command Line Options

```
--dataset DATASET      Dataset to finetune on (required)
--epochs N            Number of training epochs (default: 50)
--batch-size N        Batch size (uses dataset default if not specified)
--lr RATE             Learning rate (uses dataset default if not specified)
--remark TEXT         Experiment name for tracking (default: finetune)
--gpu N               GPU device ID to use (default: auto-detect)
--num-gpus N          Number of GPUs for distributed training (default: 1)
--skip-download       Skip downloading model weights and dataset
--skip-prepare        Skip dataset preparation
--list-datasets       List all available datasets
```

## Examples

### Basic Finetuning
```bash
# Finetune on Epilepsy dataset
uv run python finetune_cli.py --dataset Epilepsy

# Finetune on EMG dataset with 100 epochs
uv run python finetune_cli.py --dataset EMG --epochs 100
```

### Advanced Usage
```bash
# Use specific GPU
uv run python finetune_cli.py --dataset Gesture --gpu 0

# Multi-GPU training
uv run python finetune_cli.py --dataset FD --num-gpus 4

# Custom hyperparameters
uv run python finetune_cli.py --dataset Epilepsy --epochs 75 --batch-size 16 --lr 0.0001
```

### Development Workflow
```bash
# Skip downloads if you already have the data
uv run python finetune_cli.py --dataset EMG --skip-download --skip-prepare

# Add experiment tracking name
uv run python finetune_cli.py --dataset Epilepsy --remark experiment_v2
```

## Expected Runtime

Approximate training times (on CPU):
- **Epilepsy**: ~12 minutes per epoch (60 train samples)
- **EMG**: ~5 minutes per epoch 
- **FD**: ~3 minutes per epoch
- **Gesture**: ~8 minutes per epoch
- **SleepEEG**: ~10 minutes per epoch

GPU training is typically 5-10x faster.

## Output

The finetuning results will be saved in:
```
data/results/Normwear_Downstream/{DATASET_NAME}/{TIMESTAMP}/
```

This includes:
- Model checkpoints
- Training logs
- TensorBoard logs (in the `log/` subdirectory)
- Performance metrics (AUC, accuracy, F1, etc.)

## Troubleshooting

1. **CUDA out of memory**: Reduce batch size
   ```bash
   uv run python finetune_cli.py --dataset Epilepsy --batch-size 8
   ```

2. **Dataset already exists**: Use `--skip-download` to skip re-downloading
   ```bash
   uv run python finetune_cli.py --dataset EMG --skip-download
   ```

3. **Missing dependencies**: Run uv sync
   ```bash
   uv sync
   ```

4. **uv not installed**: Install uv first
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

## Adding New Datasets

To add a new dataset:

1. Add dataset info to `AVAILABLE_DATASETS` in `finetune_cli.py`
2. Create a conversion script in `data_converters/` (e.g., `data_converters/convert_newdataset_data.py`)
3. Add dataset configuration to `downstream_pipeline/config.py`

## Notes

- The script automatically detects if CUDA is available
- Dataset preparation is cached - it won't re-process if files exist
- Model weights are downloaded to `weights/normwear_pretrained.pth`
- Raw datasets are downloaded to `data/{DATASET_NAME}/`
- Processed datasets are stored in `data/{dataset_name}/sample_for_downstream/`