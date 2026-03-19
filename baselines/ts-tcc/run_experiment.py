#!/usr/bin/env python3
"""
TS-TCC Experiment Runner
A seamless CLI experience for running transfer learning experiments.

Available experiments:
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
import time


class ExperimentRunner:
    """Manages the execution of TS-TCC experiments."""
    
    EXPERIMENTS = {
        "sleepeeg_to_epilepsy": {
            "name": "SleepEEG to Epilepsy Transfer",
            "pretrain_dataset": "SleepEEG",
            "finetune_dataset": "Epilepsy",
            "required_datasets": ["SleepEEG", "Epilepsy"]
        },
        "ecg_to_emg": {
            "name": "ECG to EMG Transfer",
            "pretrain_dataset": "ECG",
            "finetune_dataset": "EMG",
            "required_datasets": ["ECG", "EMG"]
        },
        "fda_to_fdb": {
            "name": "FD-A to FD-B Transfer",
            "pretrain_dataset": "FD_A",
            "finetune_dataset": "FD_B",
            "required_datasets": ["FD_A", "FD_B"]
        },
        "har_to_gesture": {
            "name": "HAR to Gesture Transfer",
            "pretrain_dataset": "HAR",
            "finetune_dataset": "Gesture",
            "required_datasets": ["HAR", "Gesture"]
        }
    }
    
    def __init__(self):
        self.root_dir = Path.cwd()
        self.data_dir = self.root_dir / "src" / "ts_tcc" / "data"
        self.logs_dir = self.root_dir / "src" / "ts_tcc" / "experiments_logs"
        
    def check_dataset_exists(self, dataset_name):
        """Check if a dataset exists in the data directory."""
        dataset_path = self.data_dir / dataset_name
        if not dataset_path.exists():
            return False
        
        # Check for required files
        required_files = ["train.pt", "val.pt", "test.pt"]
        for file in required_files:
            if not (dataset_path / file).exists():
                return False
        return True
    
    def download_datasets(self, force=False):
        """Download all datasets using the download script."""
        print("\n" + "="*60)
        print("Dataset Download")
        print("="*60)
        
        # Check if datasets already exist
        all_datasets = ["SleepEEG", "Epilepsy", "FD_A", "FD_B", "HAR", "Gesture", "ECG", "EMG"]
        missing_datasets = [d for d in all_datasets if not self.check_dataset_exists(d)]
        
        if not missing_datasets and not force:
            print("✓ All datasets are already downloaded.")
            return True
        
        if missing_datasets:
            print(f"Missing datasets: {', '.join(missing_datasets)}")
        
        response = input("\nDo you want to download datasets? [y/N]: ").strip().lower()
        if response != 'y':
            print("Skipping dataset download.")
            return False
        
        print("\nDownloading datasets...")
        try:
            subprocess.run(["bash", "download_datasets.sh"], check=True)
            print("✓ Datasets downloaded successfully!")
            return True
        except subprocess.CalledProcessError as e:
            print(f"✗ Error downloading datasets: {e}")
            return False
    
    def check_experiment_requirements(self, experiment_key):
        """Check if required datasets exist for an experiment."""
        experiment = self.EXPERIMENTS[experiment_key]
        missing = []
        
        for dataset in experiment["required_datasets"]:
            if not self.check_dataset_exists(dataset):
                missing.append(dataset)
        
        if missing:
            print(f"\n✗ Missing required datasets: {', '.join(missing)}")
            print("Please run with --download flag to download datasets.")
            return False
        
        return True
    
    def run_command(self, cmd, description, env=None):
        """Run a command and display progress."""
        print(f"\n{description}...")
        print(f"Command: {' '.join(cmd)}")
        print("-" * 60)
        
        start_time = time.time()
        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env=env
            )
            
            # Stream output in real-time
            for line in process.stdout:
                print(line, end='')
            
            process.wait()
            
            if process.returncode == 0:
                elapsed = time.time() - start_time
                print(f"\n✓ {description} completed successfully! (Time: {elapsed:.1f}s)")
                return True
            else:
                print(f"\n✗ {description} failed with return code {process.returncode}")
                return False
                
        except Exception as e:
            print(f"\n✗ Error during {description}: {e}")
            return False
    
    def run_experiment(self, experiment_key, seed=42, device="cpu", skip_validation=True):
        """Run a complete experiment (pretrain + finetune)."""
        experiment = self.EXPERIMENTS[experiment_key]
        
        print("\n" + "="*60)
        print(f"Running Experiment: {experiment['name']}")
        print("="*60)
        
        # Check requirements
        if not self.check_experiment_requirements(experiment_key):
            return False
        
        # Create logs directory if it doesn't exist
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        
        # Set PYTHONPATH to include src directory
        env = os.environ.copy()
        env['PYTHONPATH'] = str(self.root_dir / "src") + ":" + env.get('PYTHONPATH', '')
        
        # Pretrain command
        pretrain_cmd = [
            sys.executable,
            "-m", "ts_tcc.main",
            f"--experiment_description={experiment_key}_transfer",
            "--run_description=pretrain",
            f"--seed={seed}",
            "--training_mode=self_supervised",
            f"--selected_dataset={experiment['pretrain_dataset']}",
            f"--logs_save_dir={self.logs_dir}",
            f"--device={device}",
        ]
        
        if skip_validation:
            pretrain_cmd.append("--skip_validation")
        
        # Run pretraining
        if not self.run_command(pretrain_cmd, "Pretraining Step", env=env):
            print("\n✗ Pretraining failed. Aborting experiment.")
            return False
        
        # Finetune command
        finetune_cmd = [
            sys.executable,
            "-m", "ts_tcc.main",
            f"--experiment_description={experiment_key}_transfer",
            "--run_description=finetune",
            f"--seed={seed}",
            "--training_mode=fine_tune",
            f"--selected_dataset={experiment['finetune_dataset']}",
            f"--logs_save_dir={self.logs_dir}",
            f"--device={device}",
        ]
        
        if skip_validation:
            finetune_cmd.append("--skip_validation")
        
        # Run finetuning
        if not self.run_command(finetune_cmd, "Finetuning Step", env=env):
            print("\n✗ Finetuning failed.")
            return False
        
        print("\n" + "="*60)
        print(f"✓ Experiment '{experiment['name']}' completed successfully!")
        print(f"Results saved in: {self.logs_dir / f'{experiment_key}_transfer'}")
        print("="*60)
        
        return True
    
    def list_experiments(self):
        """List all available experiments."""
        print("\nAvailable experiments:")
        print("-" * 40)
        for key, exp in self.EXPERIMENTS.items():
            print(f"  {key:<20} - {exp['name']}")
        print()


def main():
    """Main entry point for the experiment runner."""
    parser = argparse.ArgumentParser(
        description="TS-TCC Experiment Runner - Seamless CLI for transfer learning experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run SleepEEG to Epilepsy experiment
  python run_experiment.py sleepeeg_to_epilepsy
  
  # Download datasets and run experiment
  python run_experiment.py sleepeeg_to_epilepsy --download
  
  # Run with GPU
  python run_experiment.py sleepeeg_to_epilepsy --device cuda
  
  # List all available experiments
  python run_experiment.py --list
        """
    )
    
    parser.add_argument(
        "experiment",
        nargs="?",
        choices=["sleepeeg_to_epilepsy", "ecg_to_emg", "fda_to_fdb", "har_to_gesture"],
        help="Experiment to run"
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download datasets before running experiment"
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
        help="Device to use for training (default: cpu)"
    )
    parser.add_argument(
        "--validation",
        action="store_true",
        help="Run validation during training (default: skip validation)"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all available experiments"
    )
    
    args = parser.parse_args()
    
    runner = ExperimentRunner()
    
    # List experiments if requested
    if args.list:
        runner.list_experiments()
        return
    
    # Check if experiment is provided
    if not args.experiment:
        parser.print_help()
        runner.list_experiments()
        return
    
    # Welcome message
    print("\n" + "="*60)
    print("TS-TCC Experiment Runner")
    print("="*60)
    
    # Download datasets if requested
    if args.download:
        if not runner.download_datasets():
            print("\n✗ Dataset download failed or cancelled.")
            sys.exit(1)
    
    # Run the experiment
    success = runner.run_experiment(
        args.experiment,
        seed=args.seed,
        device=args.device,
        skip_validation=not args.validation
    )
    
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()