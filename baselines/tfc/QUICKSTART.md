# Quick Start Guide

Get up and running with TF-C experiments in under 5 minutes!

## Prerequisites

- Python 3.12+
- [uv](https://github.com/astral-sh/uv) package manager
- ~500MB free disk space for datasets

## 1. Clone and Setup (30 seconds)

```bash
git clone <repository-url>
cd tf-c-test
uv sync
```

## 2. Run Your First Experiment

```bash
uv run python run_experiment.py --experiment sleepeeg_to_epilepsy
```

This runs a complete SleepEEG → Epilepsy transfer learning experiment.

## What Just Happened?

1. **Downloaded datasets**: SleepEEG (439MB) and Epilepsy (16MB)
2. **Pre-trained**: Self-supervised learning on SleepEEG data (~19 min)
3. **Fine-tuned**: Supervised learning on Epilepsy data (~7 sec)
4. **Evaluated**: Tested on Epilepsy test set (82.89% accuracy with KNN)

## Next Steps

### Run Other Experiments

```bash
# ECG to EMG transfer
uv run python run_experiment.py --experiment ecg_to_emg

# Human Activity Recognition to Gesture
uv run python run_experiment.py --experiment har_to_gesture

# Fault Detection A to B
uv run python run_experiment.py --experiment fda_to_fdb
```

### Use GPU (if available)

```bash
uv run python run_experiment.py --experiment sleepeeg_to_epilepsy --device cuda
```

### Run Full Training (not just 1 epoch)

1. Edit `src/tfc/config_files/SleepEEG_Configs.py`:
   ```python
   self.num_epoch = 10  # Change from 1
   self.num_epoch_finetune = 100  # Change from 1
   ```

2. Edit `src/tfc/config_files/Epilepsy_Configs.py`:
   ```python
   self.num_epoch = 40  # Change from 1
   ```

3. Run the experiment again:
   ```bash
   uv run python run_experiment.py --experiment sleepeeg_to_epilepsy
   ```

## Common Commands

```bash
# Download all datasets upfront
bash download_datasets.sh

# Run only pre-training
uv run python run_experiment.py --experiment sleepeeg_to_epilepsy --steps pretrain

# Run only fine-tuning (requires pre-trained model)
uv run python run_experiment.py --experiment sleepeeg_to_epilepsy --steps finetune

# Check results
ls -la ./src/tfc/experiments_logs/
```

## Tips

- Start with 1 epoch to verify everything works
- Use CPU for testing, GPU for real experiments
- Each experiment creates a unique folder with logs and models
- Pre-trained models can be reused for multiple fine-tuning runs

Happy experimenting! 🚀