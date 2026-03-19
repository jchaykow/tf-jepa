# TS-TCC Test Repository

This repository contains the Time-Series Representation Learning via Temporal and Contextual Contrasting (TS-TCC) implementation with streamlined CLI tools for running transfer learning experiments.

## Quick Start

### Option 1: Automated Setup (Recommended)

```bash
# Run the setup script - it will install uv, dependencies, and optionally download datasets
./setup.sh
```

### Option 2: Manual Setup

#### 1. Install uv (package manager)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

#### 2. Install Dependencies

```bash
uv add torch numpy scikit-learn pandas tqdm einops openpyxl
```

#### 3. Download Datasets

```bash
bash download_datasets.sh
```

### Run an Experiment

```bash
# Using Python script (recommended)
uv run python run_experiment.py sleepeeg_to_epilepsy

# Or using Bash script
./run_experiment.sh sleepeeg_to_epilepsy
```

## Available Experiments

- `sleepeeg_to_epilepsy`: SleepEEG → Epilepsy transfer learning
- `ecg_to_emg`: ECG → EMG transfer learning
- `fda_to_fdb`: FD-A → FD-B transfer learning
- `har_to_gesture`: HAR → Gesture transfer learning

## CLI Tools

This repository provides two convenient ways to run experiments:

1. **Python Script** (`run_experiment.py`): Full-featured experiment runner with progress tracking
2. **Bash Script** (`run_experiment.sh`): Simple shell script for quick experiments

Both tools handle the complete experiment workflow:
- Dataset validation
- Pretraining on source dataset
- Finetuning on target dataset
- Results logging

## Documentation

- **CLI Usage Guide**: See [CLI_USAGE.md](CLI_USAGE.md) for detailed instructions
- **Original TS-TCC README**: See [src/ts_tcc/README.md](src/ts_tcc/README.md) for model details

## Example Commands

```bash
# List available experiments
uv run python run_experiment.py --list

# Run with GPU
uv run python run_experiment.py ecg_to_emg --device cuda

# Download datasets and run
uv run python run_experiment.py har_to_gesture --download

# Run with custom seed
./run_experiment.sh fda_to_fdb --seed 123
```

## Important Notes

1. **Dependencies**: The project requires PyTorch and several other packages. Use `uv` to manage dependencies as shown above.
2. **Dataset Downloads**: The `download_datasets.sh` script has been updated with User-Agent headers to work with Figshare.
3. **Data Path**: The hardcoded data path in `src/ts_tcc/main.py` has been fixed to use relative paths.
4. **Python Path**: The experiment runner automatically sets up the Python path to find the `ts_tcc` module.

## Results

Experiment results are saved in `src/ts_tcc/experiments_logs/` with detailed logs and performance metrics.

## Citation

If you use this code, please cite the original TS-TCC paper:

```bibtex
@inproceedings{ijcai2021-324,
  title     = {Time-Series Representation Learning via Temporal and Contextual Contrasting},
  author    = {Eldele, Emadeldeen and Ragab, Mohamed and Chen, Zhenghua and Wu, Min and Kwoh, Chee Keong and Li, Xiaoli and Guan, Cuntai},
  booktitle = {Proceedings of the Thirtieth International Joint Conference on Artificial Intelligence, {IJCAI-21}},
  pages     = {2352--2359},
  year      = {2021},
}
```
