# NormWear Finetuning Tools

This repository includes easy-to-use CLI tools for finetuning the NormWear model on various physiological signal datasets.

## 🚀 Quick Start for New Developers

### Option 1: One-Command Setup and Test (Recommended)
```bash
# This will set up everything and run a quick test
./quickstart.sh
```

### Option 2: Step-by-Step Setup
```bash
# 1. Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Set up the environment
uv sync

# 3. Run finetuning (downloads model & data automatically)
uv run python finetune_cli.py --dataset Epilepsy
```

## 📋 Available Tools

### 1. **finetune_cli.py** - Main Finetuning CLI
A comprehensive CLI tool that handles the entire finetuning workflow:
- Downloads pretrained model weights automatically
- Downloads and prepares datasets
- Runs finetuning with configurable parameters
- Supports multiple datasets and GPU training

### 2. **setup_environment.sh** - Environment Setup
Sets up the Python environment with all required dependencies.

### 3. **quickstart.sh** - All-in-One Quick Start
Perfect for new developers - sets up everything and runs a test in one command.

## 📊 Supported Datasets

| Dataset | Description | Channels | Classes | Task |
|---------|-------------|----------|---------|------|
| Epilepsy | EEG seizure detection | 1 | 2 | Classification |
| EMG | Electromyography | 1 | 3 | Classification |
| FD | Fault Detection | 1 | 3 | Classification |
| Gesture | Motion sensor gestures | 3 | 8 | Classification |
| SleepEEG | Sleep stage classification | 1 | 5 | Classification |

## 🎯 Common Usage Examples

```bash
# List all available datasets
uv run python finetune_cli.py --list-datasets

# Quick 2-epoch test run
uv run python finetune_cli.py --dataset Epilepsy --epochs 2

# Full training with custom settings
uv run python finetune_cli.py --dataset EMG --epochs 100 --batch-size 32 --lr 0.001

# Multi-GPU training
uv run python finetune_cli.py --dataset Gesture --num-gpus 4

# Use specific GPU
uv run python finetune_cli.py --dataset FD --gpu 0
```

## 📁 Project Structure

```
normwear-test/
├── finetune_cli.py          # Main CLI tool
├── setup_environment.sh     # Environment setup script
├── quickstart.sh           # Quick start script
├── FINETUNE_GUIDE.md       # Detailed finetuning guide
├── data_converters/        # Dataset conversion scripts
│   ├── convert_epilepsy_data.py
│   ├── convert_emg_data.py
│   ├── convert_fd_data.py
│   ├── convert_gesture_data.py
│   └── convert_sleepeeg_data.py
├── data/                   # Downloaded datasets
├── weights/                # Model weights
├── downstream_pipeline/    # Finetuning implementation
└── modules/               # Model architecture
```

## 📚 Documentation

- **[FINETUNE_GUIDE.md](FINETUNE_GUIDE.md)** - Comprehensive guide with all options and troubleshooting
- Run `uv run python finetune_cli.py --help` for command-line help

## ⚡ Performance

- **CPU**: ~12 minutes per epoch for Epilepsy dataset (60 samples)
- **GPU**: 5-10x faster than CPU
- Model weights: ~500MB download (one-time)
- Dataset sizes: 50MB-500MB depending on dataset

## 🛠️ Requirements

- Python 3.12+
- uv (Python package manager) - [Install here](https://github.com/astral-sh/uv)
- ~10GB free disk space
- (Optional) CUDA-capable GPU for faster training

## 🤝 Contributing

To add a new dataset:
1. Add dataset info to `AVAILABLE_DATASETS` in `finetune_cli.py`
2. Create a conversion script in `data_converters/` (e.g., `data_converters/convert_newdataset_data.py`)
3. Add configuration to `downstream_pipeline/config.py`

## 📝 License

See [LICENSE](LICENSE) file for details.