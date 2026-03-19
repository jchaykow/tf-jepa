# Running TF-C Transfer Learning Experiments

This guide explains how to run transfer learning experiments using the TF-C (Time-Frequency Consistency) framework.

## ⚠️ Important Notes

1. **Configs are set to 1 epoch** for quick testing. For full experiments:
   - Edit `src/tfc/config_files/SleepEEG_Configs.py`: Change `num_epoch` and `num_epoch_finetune`
   - Edit `src/tfc/config_files/Epilepsy_Configs.py`: Change `num_epoch`
   - Similar changes for other dataset configs

2. **Pre-trained model fix**: The code has been updated to correctly load pre-trained models from the `pretrain` folder during fine-tuning.

## Quick Start

### Option 1: Using the Python CLI (Recommended)

The easiest way to run experiments is using the `run_experiment.py` script:

```bash
# Run the complete SleepEEG to Epilepsy experiment
uv run python run_experiment.py --experiment sleepeeg_to_epilepsy

# Run with GPU
uv run python run_experiment.py --experiment sleepeeg_to_epilepsy --device cuda

# Run only pretraining step
uv run python run_experiment.py --experiment sleepeeg_to_epilepsy --steps pretrain

# Run with custom seed
uv run python run_experiment.py --experiment sleepeeg_to_epilepsy --seed 123
```

### Option 2: Using Bash Script

For the SleepEEG to Epilepsy experiment specifically:

```bash
bash run_sleepeeg_epilepsy_experiment.sh
```

### Option 3: Direct Commands

Run the commands directly for more control:

```bash
# Step 1: Download datasets (if not already downloaded)
bash download_datasets.sh

# Step 2: Pretrain on source dataset
uv run python -m src.tfc.main \
    --run_description=pretrain \
    --seed=42 \
    --training_mode=pre_train \
    --pretrain_dataset=SleepEEG \
    --target_dataset=Epilepsy \
    --logs_save_dir=./src/tfc/experiments_logs \
    --device=cpu

# Step 3: Fine-tune on target dataset
uv run python -m src.tfc.main \
    --run_description=finetune \
    --seed=42 \
    --training_mode=fine_tune_test \
    --pretrain_dataset=SleepEEG \
    --target_dataset=Epilepsy \
    --logs_save_dir=./src/tfc/experiments_logs \
    --device=cpu
```

## Available Experiments

1. **SleepEEG to Epilepsy** (`sleepeeg_to_epilepsy`)
   - Source: Sleep EEG recordings
   - Target: Epilepsy seizure detection

2. **ECG to EMG** (`ecg_to_emg`)
   - Source: Electrocardiogram data
   - Target: Electromyography data

3. **FD-A to FD-B** (`fda_to_fdb`)
   - Source: Fault Detection dataset A
   - Target: Fault Detection dataset B

4. **HAR to Gesture** (`har_to_gesture`)
   - Source: Human Activity Recognition
   - Target: Gesture recognition

## Parameters

### Training Modes
- `pre_train`: Self-supervised pretraining on source dataset
- `fine_tune_test`: Fine-tuning and testing on target dataset

### Common Arguments
- `--seed`: Random seed for reproducibility (default: 42)
- `--device`: Computing device, either 'cpu' or 'cuda' (default: cpu)
- `--logs_save_dir`: Directory to save experiment logs (default: ./src/tfc/experiments_logs)

## Results

After running an experiment, results will be saved in:
```
./src/tfc/experiments_logs/<source>_2_<target>/<run_description>/<training_mode>_seed_<seed>_2layertransformer/
```

For example:
```
./src/tfc/experiments_logs/SleepEEG_2_Epilepsy/pretrain/pre_train_seed_42_2layertransformer/
./src/tfc/experiments_logs/SleepEEG_2_Epilepsy/finetune/fine_tune_test_seed_42_2layertransformer/
```

## Expected Results & Timing

### With 1 Epoch (Testing Configuration)

**SleepEEG to Epilepsy** on CPU:
- Pre-training: ~19 minutes, Loss: 4.79
- Fine-tuning: ~7 seconds
  - MLP Testing: Acc=80.21%, F1=44.51%, AUROC=58.75%
  - KNN Testing: Acc=82.89%, F1=57.14%, AUROC=75.05%

### With Full Epochs (Research Configuration)

Edit config files to restore default epochs:
- SleepEEG: 10 epochs pre-training, 100 epochs fine-tuning
- Epilepsy: 40 epochs
- Expect several hours of training on CPU, much faster on GPU

## Troubleshooting

1. **Missing datasets**: The scripts will automatically check for datasets and offer to download them if missing.

2. **CUDA/GPU issues**: If you get CUDA errors, try running with `--device cpu`.

3. **Memory issues**: If you run out of memory, try reducing batch size in the config files under `src/tfc/config_files/`.

4. **Import errors**: Make sure you're running from the project root directory and have all dependencies installed using `uv sync`.

5. **FileNotFoundError during fine-tuning**: This has been fixed. The code now correctly loads pre-trained models from the `pretrain` folder.