# Data Converters

This directory contains scripts for converting raw datasets into the NormWear format.

## Scripts

- **`convert_epilepsy_data.py`** - Converts Epilepsy EEG dataset
- **`convert_emg_data.py`** - Converts EMG (Electromyography) dataset
- **`convert_fd_data.py`** - Converts Fault Detection dataset
- **`convert_gesture_data.py`** - Converts Gesture recognition dataset
- **`convert_sleepeeg_data.py`** - Converts Sleep EEG dataset

## Usage

These scripts are automatically called by the `finetune_cli.py` tool when preparing datasets. You can also run them manually:

```bash
# Using uv (recommended)
uv run python data_converters/convert_epilepsy_data.py

# Or if you have activated the environment
python data_converters/convert_epilepsy_data.py
```

## Output Format

All converters output data in the following format:
- Location: `data/{dataset_name}/sample_for_downstream/`
- Format: Pickle files (`.pkl`) containing:
  - `uid`: Unique identifier
  - `data`: Numpy array of signal data (downsampled to appropriate rate)
  - `sampling_rate`: Output sampling rate (typically 64 Hz for NormWear)
  - `label`: Classification or regression labels
- Split info: `train_test_split.json` in the dataset directory

## Adding New Converters

To add a new dataset converter:

1. Create a new script: `convert_<dataset>_data.py`
2. Follow the pattern of existing converters:
   - Load raw data
   - Downsample to appropriate rate (usually 64 Hz)
   - Save as pickle files with required fields
   - Create train/test split JSON
3. Add the converter to `AVAILABLE_DATASETS` in `finetune_cli.py`
4. Add dataset configuration to `downstream_pipeline/config.py`