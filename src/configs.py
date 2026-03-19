import importlib


EXPERIMENTS = {
    "har": {
        "config_module": "config_files.HAR_Configs",
        "source_data":  "./datasets/HAR",
        "target_data":  "./datasets/Gesture",
        "model_dir":    "./models/har"
    },
    "ecg": {
        "config_module": "config_files.ECG_Configs",
        "source_data":  "./datasets/ECG",
        "target_data":  "./datasets/EMG",
        "model_dir":    "./models/ecg"
    },
    "fd": {
        "config_module": "config_files.FD_A_Configs",
        "source_data":  "./datasets/FD-A",
        "target_data":  "./datasets/FD-B",
        "model_dir":    "./models/fd"
    },
    "sleepeeg": {
        "config_module": "config_files.SleepEEG_Configs",
        "source_data":  "./datasets/SleepEEG",
        "target_data":  "./datasets/Epilepsy",
        "model_dir":    "./models/sleepeeg"
    },
}

TGT_INFO = {
    # name-key      num_classes   target_batch_size
    "Epilepsy":     (2, 60, [1, 1]),   # current default for SleepEEG → Epilepsy
    "Gesture":      (8, 128, [1, 1, 1, 1, 1, 1, 1, 1]),
    "FD-B":         (3, 128, [1, 1, 1]),
    "EMG":          (3, 61, [21, 43, 58]),
}

def load_config(module_path):
    """Dynamically import the right Config class"""
    mod = importlib.import_module(module_path)
    return mod.Config()

def to_dict(cfg):
    """Flatten the nested Config object into the dict structure expected by TF-JEPA"""
    d = {
        "input_channels": cfg.input_channels,
        "increased_dim": cfg.increased_dim,
        "final_out_channels": cfg.final_out_channels,
        "num_classes": cfg.num_classes,
        "num_classes_target": cfg.num_classes_target,
        "dropout": cfg.dropout,
        "kernel_size": cfg.kernel_size,
        "stride": cfg.stride,
        "features_len": cfg.features_len,
        "features_len_f": cfg.features_len_f,
        "TSlength_aligned": cfg.TSlength_aligned,
        "CNNoutput_channel": cfg.CNNoutput_channel,
        "num_epoch": cfg.num_epoch,
        "optimizer": cfg.optimizer,
        "beta1": cfg.beta1,
        "beta2": cfg.beta2,
        "lr": cfg.lr,
        "lr_f": cfg.lr_f,
        "drop_last": cfg.drop_last,
        "batch_size": cfg.batch_size,
        "target_batch_size": cfg.target_batch_size,
        "TFC_Loss": {
            # poly power -- only used if type == "ntxent_poly"
            "poly_power": 2,
            # reduction for the per-sample loss
            "reduction": "mean"
        },
        "use_tfc": False,
        "JEPA_Loss": {
            # choose   "cosine" | "mae" | "mse"
            "type": "cosine",
        },
        # relative weights in the final loss
        "tf_weight":   0.0,
        "jepa_weight": 1.0,
        # optional: list of parameter group names to train.
        # If absent → train everything you trained before.
        "trainable_blocks": [
            "transformer_encoder_t",
            "projector_t",
            "transformer_encoder_f_online",
            "projector_f_online",
            "online_predictor"
        ],
        "Context_Cont": {
            "temperature": cfg.Context_Cont.temperature,
            "use_cosine_similarity": cfg.Context_Cont.use_cosine_similarity
        },
        "TC": {
            "hidden_dim": cfg.TC.hidden_dim,
            "timesteps": cfg.TC.timesteps
        },
        "augmentation": {
            "jitter_scale_ratio": cfg.augmentation.jitter_scale_ratio,
            "jitter_ratio": cfg.augmentation.jitter_ratio,
            "max_seg": cfg.augmentation.max_seg
        }
    }
    return d
