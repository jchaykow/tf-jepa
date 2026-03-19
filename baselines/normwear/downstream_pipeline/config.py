MODEL_CONFIG = {
    "cv": {"ts": False, "cwt": True, "cwtp": False, "nlp": False}, 
    "chronos": {"ts": True, "cwt": False, "cwtp": False, "nlp": False}, 
    "moirai": {"ts": True, "cwt": False, "cwtp": False, "nlp": False}, 
    "pretrain": {"ts": False, "cwt": False, "cwtp": True, "nlp": True}, 
    "nlp": {"ts": False, "cwt": False, "cwtp": False, "nlp": True}, 
}

DATASET_CONFIG = {
    "wesad": {"n_ch": 10, "n_cl": 3, "task": "class", "lr": 1e-2,'bs':8,'max_len':390},
    "gameemo": {"n_ch": 4, "n_cl": 4, "task": "class", "lr": 1e-1,'bs':128},
    "uci_har": {"n_ch": 6, "n_cl": 6, "task": "class", "lr": 1e-1,'bs':32,'max_len':165},
    "non_invasive_bp": {"n_ch": 3, "n_cl": 2, "task": "reg", "lr": 1e-3,'bs':64}, 
    "ppg_hgb": {"n_ch": 2, "n_cl": 1, "task": "reg", "lr": 1e-2,'bs':8}, 
    "ecg_heart_cat": {"n_ch": 1, "n_cl": 2, "task": "class", "lr": 1e-2,'bs':128,'max_len':186},
    "PPG_HTN": {"n_ch": 1, "n_cl": 4, "task": "class", "lr": 7e-1,'bs':64,'max_len':271},
    "PPG_DM": {"n_ch": 1, "n_cl": 2, "task": "class", "lr": 1e-3,'bs':64,'max_len':271}, 
    "PPG_CVA": {"n_ch": 1, "n_cl": 2, "task": "class", "lr": 1e-4,'bs':64,'max_len':271}, 
    "PPG_CVD": {"n_ch": 1, "n_cl": 3, "task": "class",  "lr": 1e-3,'bs':128,'max_len':271},
    "maus": {"n_ch": 3, "n_cl": 2, "task": "reg",  "lr": 1e-2, 'y_range':(0,100)},
    'dalia':{'n_ch':6,'n_cl':9,'task':'class',"lr": 1e-2},
    'drive_fatigue':{'n_ch':4,'n_cl':2,'task':'class',"lr": 1e-2,'bs':16,'max_len':390}, 
    'indian-fPCG':{'n_ch':1,'n_cl':1,'task':'reg',"lr": 1e-2,'bs':2,}, #for mean pooling
    'Epilepsy':{'n_ch':1,'n_cl':2,'task':'class',"lr": 1e-2,'bs':16,'max_len':178},
    "FD": {
        "n_ch": 1,          # one input channel
        "n_cl": 3,          # adjust if you have more than two classes
        "task": "class",    # classification
        "lr": 1e-3,         # a safe default; tune later
        "bs": 8,            # fits easily on CPU; must be ≤ #train samples
        "max_len": 21       # 5120 @ 64 kHz  →  ~20 × 256 Hz after down-sampling
    },
    "Gesture": {          # <- key you’ll pass with --ds_name
        "n_ch": 3,        # three sensor channels
        "n_cl": 8,
        "task": "class",  # classification
        "lr": 1e-3,       # a safe starter learning-rate
        "bs": 32,         # fits easily on CPU/GPU; ≤ #train samples
        "max_len": 315    # sequence length (no down-sampling, 315 @ 100 Hz)
    },
    "EMG": {
        "n_ch": 1,
        "n_cl": 3,
        "task": "class",
        "lr":   1e-3,     # starter LR
        "bs":   32,       # fits on CPU; < #train samples
        "max_len": 96     # after down-sampling to 256 Hz
    },
    "emgtfc": {
        "n_ch": 1,
        "n_cl": 3,
        "task": "class",
        "lr":   1e-3,     # starter LR
        "bs":   16,       # fits on CPU; < #train samples
        "max_len": 1500
    },
    "SleepEEG": {
        "n_ch": 1,
        "n_cl": 1,       # placeholder
        "task": "class", # placeholder (ignored in SSL)
        "lr": 1e-3,
        "bs": 64,
        "max_len": 200
    },
    # Example: 'dataset_name': {'n_ch': num_of_channel, 'n_cl': num_of_class, 'task': 'class' or 'reg', 'lr': flot, 'bs': int, 'max_len': max_len_of_sequence}
}
