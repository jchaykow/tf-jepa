#!/usr/bin/env python3
"""
Seamless CLI experience for running TF-C transfer learning experiments.

This script handles the full experiment pipeline:
1. Downloads datasets if needed
2. Runs pretraining on source dataset
3. Runs fine-tuning on target dataset

Supported experiments:
- ECG to EMG
- SleepEEG to Epilepsy
- FD-A to FD-B
- HAR to Gesture
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path


# Experiment configurations
EXPERIMENTS = {
    "sleepeeg_to_epilepsy": {
        "pretrain_dataset": "SleepEEG",
        "target_dataset": "Epilepsy",
        "description": "SleepEEG to Epilepsy transfer"
    },
    "ecg_to_emg": {
        "pretrain_dataset": "ECG",
        "target_dataset": "EMG",
        "description": "ECG to EMG transfer"
    },
    "fda_to_fdb": {
        "pretrain_dataset": "FD_A",
        "target_dataset": "FD_B",
        "description": "FD-A to FD-B transfer"
    },
    "har_to_gesture": {
        "pretrain_dataset": "HAR",
        "target_dataset": "Gesture",
        "description": "HAR to Gesture transfer"
    }
}


def check_datasets_exist():
    """Check if datasets are downloaded."""
    data_dir = Path("src/tfc/data")
    required_datasets = ["SleepEEG", "Epilepsy", "FD_A", "FD_B", "HAR", "Gesture", "ECG", "EMG"]
    
    missing = []
    for dataset in required_datasets:
        dataset_path = data_dir / dataset
        if not dataset_path.exists() or not any(dataset_path.iterdir()):
            missing.append(dataset)
    
    return missing


def download_datasets():
    """Run the dataset download script."""
    print("\n📥 Downloading datasets...")
    try:
        subprocess.run(["bash", "download_datasets.sh"], check=True)
        print("✅ Datasets downloaded successfully!")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error downloading datasets: {e}")
        sys.exit(1)


def run_experiment_step(experiment_name, step, seed=42, device="cpu", skip_validation=False):
    """Run a single step of the experiment (pretrain or finetune)."""
    
    exp_config = EXPERIMENTS[experiment_name]
    
    # Build command based on step
    cmd = [
        "uv", "run",
        "python", "-m", "src.tfc.main",
        f"--run_description={step}",
        f"--seed={seed}",
        f"--device={device}",
        "--logs_save_dir=./src/tfc/experiments_logs",
    ]
    
    if step == "pretrain":
        cmd.extend([
            "--training_mode=pre_train",
            f"--pretrain_dataset={exp_config['pretrain_dataset']}",
            f"--target_dataset={exp_config['target_dataset']}",
            f"--sourcedata_path=./src/tfc/data/{exp_config['pretrain_dataset']}",
            f"--targetdata_path=./src/tfc/data/{exp_config['target_dataset']}",
        ])
    elif step == "finetune":
        cmd.extend([
            "--training_mode=fine_tune_test",
            f"--pretrain_dataset={exp_config['pretrain_dataset']}",
            f"--target_dataset={exp_config['target_dataset']}",
            f"--sourcedata_path=./src/tfc/data/{exp_config['pretrain_dataset']}",
            f"--targetdata_path=./src/tfc/data/{exp_config['target_dataset']}",
        ])
    
    # Note: The original script doesn't have a skip_validation flag,
    # but we keep the interface for future compatibility
    
    print(f"\n🚀 Running {step} step for {exp_config['description']}...")
    print(f"Command: {' '.join(cmd)}")
    
    try:
        subprocess.run(cmd, check=True)
        print(f"✅ {step.capitalize()} completed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error during {step}: {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Run TF-C transfer learning experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run SleepEEG to Epilepsy experiment with default settings
  python run_experiment.py --experiment sleepeeg_to_epilepsy
  
  # Run with GPU and custom seed
  python run_experiment.py --experiment ecg_to_emg --device cuda --seed 123
  
  # Only run pretraining step
  python run_experiment.py --experiment har_to_gesture --steps pretrain
  
  # Skip dataset download check
  python run_experiment.py --experiment fda_to_fdb --skip-download-check
        """
    )
    
    parser.add_argument(
        "--experiment",
        choices=list(EXPERIMENTS.keys()),
        required=True,
        help="Experiment to run"
    )
    
    parser.add_argument(
        "--steps",
        nargs="+",
        choices=["pretrain", "finetune", "both"],
        default=["both"],
        help="Which steps to run (default: both)"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)"
    )
    
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cpu",
        help="Device to use (default: cpu)"
    )
    
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip validation during training (currently not implemented in base script)"
    )
    
    parser.add_argument(
        "--skip-download-check",
        action="store_true",
        help="Skip checking if datasets exist"
    )
    
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Force re-download of datasets even if they exist"
    )
    
    args = parser.parse_args()
    
    # Print experiment info
    exp_config = EXPERIMENTS[args.experiment]
    print(f"\n🔬 TF-C Transfer Learning Experiment")
    print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(f"Experiment: {exp_config['description']}")
    print(f"Source Dataset: {exp_config['pretrain_dataset']}")
    print(f"Target Dataset: {exp_config['target_dataset']}")
    print(f"Device: {args.device}")
    print(f"Seed: {args.seed}")
    print(f"Steps: {', '.join(args.steps)}")
    print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")
    
    # Check and download datasets if needed
    if not args.skip_download_check:
        missing_datasets = check_datasets_exist()
        if missing_datasets or args.force_download:
            if missing_datasets:
                print(f"⚠️  Missing datasets: {', '.join(missing_datasets)}")
            if args.force_download:
                print("🔄 Force download requested")
            
            response = input("\nDownload datasets now? [Y/n]: ").strip().lower()
            if response != 'n':
                download_datasets()
            else:
                print("⚠️  Proceeding without downloading datasets...")
    
    # Determine which steps to run
    steps_to_run = []
    if "both" in args.steps:
        steps_to_run = ["pretrain", "finetune"]
    else:
        steps_to_run = args.steps
    
    # Run experiment steps
    for step in steps_to_run:
        run_experiment_step(
            args.experiment,
            step,
            seed=args.seed,
            device=args.device,
            skip_validation=args.skip_validation
        )
    
    print(f"\n✨ Experiment completed successfully!")
    print(f"📊 Results saved in: ./src/tfc/experiments_logs/")


if __name__ == "__main__":
    main()