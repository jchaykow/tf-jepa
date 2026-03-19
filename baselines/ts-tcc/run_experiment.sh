#!/bin/bash

# TS-TCC Experiment Runner - Bash Script
# A simple wrapper for running transfer learning experiments
#
# Prerequisites:
#   - uv package manager (recommended): curl -LsSf https://astral.sh/uv/install.sh | sh
#   - Dependencies: uv add torch numpy scikit-learn pandas tqdm einops openpyxl

set -e  # Exit on error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
EXPERIMENT=""
SEED=42
DEVICE="cpu"
SKIP_VALIDATION="--skip_validation"
DOWNLOAD=false

# Function to print colored output
print_color() {
    color=$1
    shift
    echo -e "${color}$@${NC}"
}

# Function to print usage
usage() {
    echo "Usage: $0 EXPERIMENT [OPTIONS]"
    echo ""
    echo "Available experiments:"
    echo "  sleepeeg_to_epilepsy   - SleepEEG to Epilepsy Transfer"
    echo "  ecg_to_emg             - ECG to EMG Transfer"
    echo "  fda_to_fdb             - FD-A to FD-B Transfer"
    echo "  har_to_gesture         - HAR to Gesture Transfer"
    echo ""
    echo "Options:"
    echo "  -s, --seed SEED        Random seed (default: 42)"
    echo "  -d, --device DEVICE    Device to use: cpu or cuda (default: cpu)"
    echo "  -v, --validation       Run with validation (default: skip validation)"
    echo "  --download             Download datasets before running"
    echo "  -h, --help             Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 sleepeeg_to_epilepsy"
    echo "  $0 sleepeeg_to_epilepsy --device cuda --seed 123"
    echo "  $0 ecg_to_emg --download"
}

# Parse arguments
if [ $# -eq 0 ]; then
    usage
    exit 1
fi

# Check for help flag first
if [ "$1" = "-h" ] || [ "$1" = "--help" ]; then
    usage
    exit 0
fi

EXPERIMENT=$1
shift

# Parse options
while [[ $# -gt 0 ]]; do
    case $1 in
        -s|--seed)
            SEED="$2"
            shift 2
            ;;
        -d|--device)
            DEVICE="$2"
            shift 2
            ;;
        -v|--validation)
            SKIP_VALIDATION=""
            shift
            ;;
        --download)
            DOWNLOAD=true
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

# Validate experiment
case $EXPERIMENT in
    sleepeeg_to_epilepsy)
        PRETRAIN_DATASET="SleepEEG"
        FINETUNE_DATASET="Epilepsy"
        ;;
    ecg_to_emg)
        PRETRAIN_DATASET="ECG"
        FINETUNE_DATASET="EMG"
        ;;
    fda_to_fdb)
        PRETRAIN_DATASET="FD_A"
        FINETUNE_DATASET="FD_B"
        ;;
    har_to_gesture)
        PRETRAIN_DATASET="HAR"
        FINETUNE_DATASET="Gesture"
        ;;
    *)
        print_color $RED "Error: Invalid experiment '$EXPERIMENT'"
        usage
        exit 1
        ;;
esac

# Header
print_color $BLUE "========================================"
print_color $BLUE "TS-TCC Experiment Runner"
print_color $BLUE "========================================"
print_color $YELLOW "Experiment: $EXPERIMENT"
print_color $YELLOW "Seed: $SEED"
print_color $YELLOW "Device: $DEVICE"
print_color $YELLOW "Validation: $([ -z "$SKIP_VALIDATION" ] && echo "Yes" || echo "No")"
print_color $BLUE "========================================"

# Download datasets if requested
if [ "$DOWNLOAD" = true ]; then
    print_color $YELLOW "\nDownloading datasets..."
    if bash download_datasets.sh; then
        print_color $GREEN "✓ Datasets downloaded successfully!"
    else
        print_color $RED "✗ Failed to download datasets"
        exit 1
    fi
fi

# Check if required datasets exist
DATA_DIR="src/ts_tcc/data"
MISSING_DATASETS=()

for dataset in $PRETRAIN_DATASET $FINETUNE_DATASET; do
    if [ ! -d "$DATA_DIR/$dataset" ] || [ ! -f "$DATA_DIR/$dataset/train.pt" ]; then
        MISSING_DATASETS+=($dataset)
    fi
done

if [ ${#MISSING_DATASETS[@]} -gt 0 ]; then
    print_color $RED "\n✗ Missing required datasets: ${MISSING_DATASETS[*]}"
    print_color $YELLOW "Please run with --download flag to download datasets."
    exit 1
fi

# Logs directory
LOGS_DIR="src/ts_tcc/experiments_logs"
mkdir -p "$LOGS_DIR"

# Set PYTHONPATH
export PYTHONPATH="${PWD}/src:$PYTHONPATH"

# Run pretraining
print_color $BLUE "\n========================================"
print_color $BLUE "Step 1: Pretraining on $PRETRAIN_DATASET"
print_color $BLUE "========================================"

# Check if uv is available
if command -v uv &> /dev/null; then
    UV_CMD="uv run python"
else
    UV_CMD="python"
    print_color $YELLOW "Warning: uv not found, using system Python"
fi

$UV_CMD -m ts_tcc.main \
    --experiment_description="${EXPERIMENT}_transfer" \
    --run_description="pretrain" \
    --seed=$SEED \
    --training_mode="self_supervised" \
    --selected_dataset="$PRETRAIN_DATASET" \
    --logs_save_dir="$LOGS_DIR" \
    --device=$DEVICE \
    $SKIP_VALIDATION

if [ $? -ne 0 ]; then
    print_color $RED "\n✗ Pretraining failed"
    exit 1
fi

print_color $GREEN "\n✓ Pretraining completed successfully!"

# Run finetuning
print_color $BLUE "\n========================================"
print_color $BLUE "Step 2: Finetuning on $FINETUNE_DATASET"
print_color $BLUE "========================================"

$UV_CMD -m ts_tcc.main \
    --experiment_description="${EXPERIMENT}_transfer" \
    --run_description="finetune" \
    --seed=$SEED \
    --training_mode="fine_tune" \
    --selected_dataset="$FINETUNE_DATASET" \
    --logs_save_dir="$LOGS_DIR" \
    --device=$DEVICE \
    $SKIP_VALIDATION

if [ $? -ne 0 ]; then
    print_color $RED "\n✗ Finetuning failed"
    exit 1
fi

# Success message
print_color $GREEN "\n========================================"
print_color $GREEN "✓ Experiment completed successfully!"
print_color $GREEN "Results saved in: $LOGS_DIR/${EXPERIMENT}_transfer"
print_color $GREEN "========================================"