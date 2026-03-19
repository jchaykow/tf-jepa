# TF-C: Time-Frequency Consistency for Transfer Learning

This repository contains the implementation of TF-C (Time-Frequency Consistency), a self-supervised learning framework for time series transfer learning.

## Overview

TF-C enables transfer learning between different time series datasets by learning representations that are consistent in both time and frequency domains. The framework supports:

- **Pre-training** on a source dataset using self-supervised learning
- **Fine-tuning** on a target dataset for downstream tasks
- **Transfer learning** between different domains (e.g., Sleep EEG → Epilepsy detection)

## Supported Transfer Tasks

1. **SleepEEG → Epilepsy**: Transfer from sleep stage classification to epilepsy detection
2. **ECG → EMG**: Transfer from electrocardiogram to electromyography
3. **FD-A → FD-B**: Transfer between fault detection datasets
4. **HAR → Gesture**: Transfer from human activity recognition to gesture recognition

## Quick Start

### Prerequisites

- Python 3.12+
- [uv](https://github.com/astral-sh/uv) package manager

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd tf-c-test
```

2. Install dependencies using uv:
```bash
uv sync
```

### Running Your First Experiment

```bash
# Run SleepEEG to Epilepsy transfer learning
uv run python run_experiment.py --experiment sleepeeg_to_epilepsy
```

This will:
- Download required datasets automatically
- Run pre-training on SleepEEG
- Run fine-tuning on Epilepsy
- Display results

**Note**: The configs are currently set to 1 epoch for quick testing. For full experiments, modify the epoch numbers in `src/tfc/config_files/`.

## Running Experiments

### Method 1: Python CLI (Recommended)

```bash
# Run complete experiment
uv run python run_experiment.py --experiment sleepeeg_to_epilepsy

# Run with GPU
uv run python run_experiment.py --experiment sleepeeg_to_epilepsy --device cuda

# Run only pre-training
uv run python run_experiment.py --experiment sleepeeg_to_epilepsy --steps pretrain

# Run only fine-tuning (requires pre-trained model)
uv run python run_experiment.py --experiment sleepeeg_to_epilepsy --steps finetune

# Other experiments
uv run python run_experiment.py --experiment ecg_to_emg
uv run python run_experiment.py --experiment fda_to_fdb
uv run python run_experiment.py --experiment har_to_gesture
```

### Method 2: Direct Commands

```bash
# Pre-training
uv run python -m src.tfc.main \
    --run_description=pretrain \
    --seed=42 \
    --training_mode=pre_train \
    --pretrain_dataset=SleepEEG \
    --target_dataset=Epilepsy \
    --sourcedata_path=./src/tfc/data/SleepEEG \
    --targetdata_path=./src/tfc/data/Epilepsy \
    --logs_save_dir=./src/tfc/experiments_logs \
    --device=cpu

# Fine-tuning
uv run python -m src.tfc.main \
    --run_description=finetune \
    --seed=42 \
    --training_mode=fine_tune_test \
    --pretrain_dataset=SleepEEG \
    --target_dataset=Epilepsy \
    --sourcedata_path=./src/tfc/data/SleepEEG \
    --targetdata_path=./src/tfc/data/Epilepsy \
    --logs_save_dir=./src/tfc/experiments_logs \
    --device=cpu
```

## Datasets

### Automatic Download

Datasets are automatically downloaded when you run an experiment. You can also manually download them:

```bash
# Download all datasets
bash download_datasets.sh

# Download only SleepEEG and Epilepsy
bash download_sleepeeg_epilepsy.sh
```

### Dataset Locations

Downloaded datasets are stored in `src/tfc/data/`:
- `src/tfc/data/SleepEEG/`
- `src/tfc/data/Epilepsy/`
- `src/tfc/data/ECG/`
- `src/tfc/data/EMG/`
- etc.

## Configuration

### Modifying Training Parameters

Edit the config files in `src/tfc/config_files/`:
- `SleepEEG_Configs.py`: Pre-training epochs, batch size, learning rate
- `Epilepsy_Configs.py`: Fine-tuning epochs, model architecture
- Similar files for other datasets

### Key Parameters

```python
# In config files
self.num_epoch = 10  # Pre-training epochs
self.num_epoch_finetune = 100  # Fine-tuning epochs
self.batch_size = 128
self.lr = 3e-4
```

## Results

Results are saved in `./src/tfc/experiments_logs/` with the following structure:

```
experiments_logs/
└── SleepEEG_2_Epilepsy/
    ├── pretrain/
    │   └── pre_train_seed_42_2layertransformer/
    │       ├── logs_*.log
    │       └── saved_models/
    │           └── ckp_last.pt
    └── finetune/
        └── fine_tune_test_seed_42_2layertransformer/
            └── logs_*.log
```

### Example Results (1 epoch test)

- **Pre-training**: ~19 minutes on CPU, Loss: 4.79
- **Fine-tuning**: ~7 seconds on CPU
  - MLP Accuracy: 80.21%
  - KNN Accuracy: 82.89%
  - KNN F1: 57.14%

## Troubleshooting

### Common Issues

1. **CUDA/GPU errors**: Use `--device cpu` if GPU is not available
2. **Memory issues**: Reduce batch size in config files
3. **Missing datasets**: Scripts will auto-download, or run `download_datasets.sh`
4. **Import errors**: Ensure you're using `uv run` to execute scripts

### Fixed Issues

- ✅ Fine-tuning now correctly loads pre-trained models from the `pretrain` folder
- ✅ Dataset paths are properly configured for the expected directory structure

## Development

### Project Structure

```
tf-c-test/
├── src/tfc/
│   ├── main.py              # Main training script
│   ├── model.py             # TF-C model architecture
│   ├── trainer.py           # Training logic
│   ├── dataloader.py        # Data loading utilities
│   ├── config_files/        # Dataset-specific configs
│   └── data/                # Downloaded datasets
├── run_experiment.py        # High-level experiment runner
└── download_datasets.sh     # Dataset download script
```

### Adding New Transfer Tasks

1. Create config files for source and target datasets in `src/tfc/config_files/`
2. Add dataset download URLs to `download_datasets.sh`
3. Update `EXPERIMENTS` dict in `run_experiment.py`

## Citation

If you use this code in your research, please cite the original TF-C paper.

## License

[Add license information here]
