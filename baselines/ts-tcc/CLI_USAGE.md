# TS-TCC CLI Usage Guide

This guide provides instructions for running TS-TCC transfer learning experiments using our streamlined CLI tools.

## Prerequisites

### Install Dependencies

This project uses `uv` for dependency management. If you don't have `uv` installed:

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then install the project dependencies:

```bash
# Install all required packages
uv add torch numpy scikit-learn pandas tqdm einops openpyxl
```

## Quick Start

### Running an Experiment

The simplest way to run an experiment is:

```bash
# Using Python script with uv
uv run python run_experiment.py sleepeeg_to_epilepsy

# Using Bash script
./run_experiment.sh sleepeeg_to_epilepsy
```

This will:
1. Check if required datasets are available
2. Run the pretraining step on the source dataset
3. Run the finetuning step on the target dataset
4. Save results in `src/ts_tcc/experiments_logs/`

## Available Experiments

| Experiment Code | Description | Source Dataset | Target Dataset |
|----------------|-------------|----------------|----------------|
| `sleepeeg_to_epilepsy` | SleepEEG to Epilepsy Transfer | SleepEEG | Epilepsy |
| `ecg_to_emg` | ECG to EMG Transfer | ECG | EMG |
| `fda_to_fdb` | FD-A to FD-B Transfer | FD_A | FD_B |
| `har_to_gesture` | HAR to Gesture Transfer | HAR | Gesture |

## Command Line Options

### Python Script (`run_experiment.py`)

```bash
uv run python run_experiment.py [experiment] [options]
```

**Options:**
- `--download`: Download datasets before running experiment
- `--seed SEED`: Set random seed (default: 42)
- `--device {cpu,cuda}`: Choose device for training (default: cpu)
- `--validation`: Run with validation (default: skip validation for faster training)
- `--list`: List all available experiments

**Examples:**
```bash
# Download datasets and run experiment
uv run python run_experiment.py sleepeeg_to_epilepsy --download

# Run with GPU and custom seed
uv run python run_experiment.py ecg_to_emg --device cuda --seed 123

# Run with validation enabled
uv run python run_experiment.py har_to_gesture --validation

# List available experiments
uv run python run_experiment.py --list
```

### Bash Script (`run_experiment.sh`)

```bash
./run_experiment.sh EXPERIMENT [options]
```

**Options:**
- `-s, --seed SEED`: Set random seed (default: 42)
- `-d, --device DEVICE`: Choose device (cpu or cuda, default: cpu)
- `-v, --validation`: Run with validation
- `--download`: Download datasets before running
- `-h, --help`: Show help message

**Examples:**
```bash
# Basic run
./run_experiment.sh sleepeeg_to_epilepsy

# With GPU and custom seed
./run_experiment.sh ecg_to_emg --device cuda --seed 123

# Download datasets first
./run_experiment.sh fda_to_fdb --download
```

## Dataset Management

### Automatic Download

Both scripts can automatically download datasets when needed:

```bash
# Python script
uv run python run_experiment.py sleepeeg_to_epilepsy --download

# Bash script
./run_experiment.sh sleepeeg_to_epilepsy --download
```

**Note**: The download script has been updated to include proper User-Agent headers for Figshare compatibility.

### Manual Download

You can also download all datasets manually:

```bash
bash download_datasets.sh
```

### Dataset Structure

Datasets are stored in `src/ts_tcc/data/` with the following structure:
```
src/ts_tcc/data/
├── SleepEEG/
│   ├── train.pt
│   ├── val.pt
│   └── test.pt
├── Epilepsy/
│   ├── train.pt
│   ├── val.pt
│   └── test.pt
└── ... (other datasets)
```

## Experiment Workflow

Each experiment consists of two phases:

### 1. Pretraining Phase
- Trains the model using self-supervised learning on the source dataset
- Saves the pretrained model in the experiments logs directory

### 2. Finetuning Phase
- Loads the pretrained model
- Finetunes it on the target dataset for the classification task
- Generates final performance metrics

## Output and Results

Results are saved in `src/ts_tcc/experiments_logs/` with the following structure:

```
src/ts_tcc/experiments_logs/
└── sleepeeg_to_epilepsy_transfer/
    ├── pretrain/
    │   ├── logs.log
    │   ├── model_files/
    │   └── ...
    └── finetune/
        ├── logs.log
        ├── classification_report.xlsx
        └── ...
```

## Direct Command Execution

For advanced users who want more control, you can run the commands directly:

### Pretraining
```bash
# Set PYTHONPATH to include src directory
export PYTHONPATH="${PWD}/src:$PYTHONPATH"

# Run pretraining
uv run python -m ts_tcc.main \
    --experiment_description=sleepeeg_to_epilepsy_transfer \
    --run_description=pretrain \
    --seed=42 \
    --training_mode=self_supervised \
    --selected_dataset=SleepEEG \
    --logs_save_dir=src/ts_tcc/experiments_logs \
    --device=cpu \
    --skip_validation
```

### Finetuning
```bash
# Set PYTHONPATH to include src directory
export PYTHONPATH="${PWD}/src:$PYTHONPATH"

# Run finetuning
uv run python -m ts_tcc.main \
    --experiment_description=sleepeeg_to_epilepsy_transfer \
    --run_description=finetune \
    --seed=42 \
    --training_mode=fine_tune \
    --selected_dataset=Epilepsy \
    --logs_save_dir=src/ts_tcc/experiments_logs \
    --device=cpu \
    --skip_validation
```

## Troubleshooting

### Missing Datasets
If you see an error about missing datasets:
1. Run with the `--download` flag
2. Or manually run `bash download_datasets.sh`

### CUDA/GPU Issues
If you encounter GPU-related errors:
1. Ensure PyTorch is installed with CUDA support
2. Check your CUDA version compatibility
3. Fall back to CPU with `--device cpu`

### Memory Issues
For large datasets or limited memory:
1. Use CPU instead of GPU
2. Check the batch size configurations in `src/ts_tcc/config_files/`
3. Enable validation to monitor memory usage

## Tips for Best Results

1. **Reproducibility**: Use the same seed value across runs for consistent results
2. **GPU Usage**: Use `--device cuda` for faster training if you have a compatible GPU
3. **Validation**: Enable validation with `--validation` flag to monitor training progress
4. **Multiple Runs**: Run experiments with different seeds (e.g., 2, 3, 4, 5, 123) for statistical significance

## Example Full Workflow

Here's a complete example workflow for a new user:

```bash
# 1. Clone the repository (if not already done)
git clone <repository_url>
cd ts-tcc-test

# 2. Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 3. Install dependencies
uv add torch numpy scikit-learn pandas tqdm einops openpyxl

# 4. Download all datasets
bash download_datasets.sh

# 5. Run the SleepEEG to Epilepsy experiment
uv run python run_experiment.py sleepeeg_to_epilepsy --device cpu --seed 42

# 6. Check results
ls src/ts_tcc/experiments_logs/sleepeeg_to_epilepsy_transfer/

# 7. Run another experiment
uv run python run_experiment.py ecg_to_emg --device cuda --seed 123
```

## Additional Resources

- Main README: `src/ts_tcc/README.md`
- Configuration files: `src/ts_tcc/config_files/`
- Original paper: [IJCAI 2021](https://www.ijcai.org/proceedings/2021/0324.pdf)