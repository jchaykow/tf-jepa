import logging
import os
import random
import sys
from shutil import copy

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
)


def set_requires_grad(model, dict_, requires_grad=True):
    for param in model.named_parameters():
        if param[0] in dict_:
            param[1].requires_grad = requires_grad


def fix_randomness(SEED):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def _calc_metrics(pred_labels, true_labels, log_dir, home_path):
    pred_labels = np.array(pred_labels).astype(int)
    true_labels = np.array(true_labels).astype(int)

    # Calculate metrics
    r = classification_report(true_labels, pred_labels, digits=6, output_dict=True)
    cm = confusion_matrix(true_labels, pred_labels)
    df = pd.DataFrame(r)
    df["cohen"] = cohen_kappa_score(true_labels, pred_labels)
    df["accuracy"] = accuracy_score(true_labels, pred_labels)
    df = df * 100

    # Display results
    print("\n" + "=" * 50)
    print("CLASSIFICATION RESULTS")
    print("=" * 50)
    print(f"Accuracy: {df['accuracy'].iloc[0]:.2f}%")
    print(f"Cohen's Kappa: {df['cohen'].iloc[0]:.4f}")
    print("\nDetailed Classification Report:")
    print(df.round(2))
    print("\nConfusion Matrix:")
    print(cm)
    print("=" * 50)


def _logger(logger_name, level=logging.DEBUG):
    """
    Method to return a custom logger with the given name and level
    """
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)
    # format_string = ("%(asctime)s — %(name)s — %(levelname)s — %(funcName)s:"
    #                 "%(lineno)d — %(message)s")
    format_string = "%(message)s"
    log_format = logging.Formatter(format_string)
    # Creating and adding the console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_format)
    logger.addHandler(console_handler)
    # Creating and adding the file handler
    file_handler = logging.FileHandler(logger_name, mode="a")
    file_handler.setFormatter(log_format)
    logger.addHandler(file_handler)
    return logger


def copy_Files(destination, data_type):
    """
    Copy source files to destination directory for backup/reproducibility.
    Uses package installation path when running as installed wheel.
    """
    destination_dir = os.path.join(destination, "model_files")
    os.makedirs(destination_dir, exist_ok=True)

    try:
        # Try to find the package installation path
        import ts_tcc

        package_dir = os.path.dirname(ts_tcc.__file__)

        # Copy files using absolute paths from the installed package
        copy(os.path.join(package_dir, "main.py"), os.path.join(destination_dir, "main.py"))
        copy(
            os.path.join(package_dir, "trainer", "trainer.py"),
            os.path.join(destination_dir, "trainer.py"),
        )
        copy(
            os.path.join(package_dir, "config_files", f"{data_type}_Configs.py"),
            os.path.join(destination_dir, f"{data_type}_Configs.py"),
        )
        copy(
            os.path.join(package_dir, "dataloader", "augmentations.py"),
            os.path.join(destination_dir, "augmentations.py"),
        )
        copy(
            os.path.join(package_dir, "dataloader", "dataloader.py"),
            os.path.join(destination_dir, "dataloader.py"),
        )
        copy(
            os.path.join(package_dir, "models", "model.py"),
            os.path.join(destination_dir, "model.py"),
        )
        copy(
            os.path.join(package_dir, "models", "loss.py"), os.path.join(destination_dir, "loss.py")
        )
        copy(os.path.join(package_dir, "models", "TC.py"), os.path.join(destination_dir, "TC.py"))

    except Exception as e:
        # If copying fails (e.g., files not found), log the error but don't stop execution
        print(f"Warning: Could not copy source files for backup: {e}")
        print("This is expected when running as an installed package. Continuing without backup.")
