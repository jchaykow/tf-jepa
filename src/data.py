import torch
from torch.utils.data import Dataset
import os
import numpy as np
import torch.fft as fft
from src.utils import DataTransform_TD, DataTransform_FD


class Load_Dataset(Dataset):
    def __init__(self, tensor_dict, cfg, mode):
        self.x = tensor_dict["samples"]           # torch tensors expected
        self.y = tensor_dict["labels"].long()
        if self.x.ndim == 2:                      # (B,T) → (B,1,T)
            self.x = self.x.unsqueeze(1)
        self.x = self.x[:, :1, :cfg["TSlength_aligned"]]  # crop
        self.mode = mode
        self.cfg = cfg

    def _augment_time(self, x):
        # cheap on-the-fly jitter/scale/permute
        return DataTransform_TD(x, self.cfg)

    def _augment_freq(self, xf):
        return DataTransform_FD(xf, self.cfg)

    def __getitem__(self, idx):
        x_t  = self.x[idx]
        xf   = fft.fft(x_t).abs()                 # compute per-sample
        if self.mode == "pre_train":
            x_t_aug  = self._augment_time(x_t)
            xf_aug   = self._augment_freq(xf)
            return x_t, self.y[idx], x_t_aug, xf, xf_aug
        else:                                     # finetune / test
            return x_t, self.y[idx], x_t, xf, xf

    def __len__(self):
        return len(self.x)


def data_generator(sourcedata_path, targetdata_path, configs, training_mode, subset=True):
    train_dataset = torch.load(os.path.join(sourcedata_path, "train.pt"), weights_only=True)
    finetune_dataset = torch.load(os.path.join(targetdata_path, "train.pt"), weights_only=True)  # train.pt
    test_dataset = torch.load(os.path.join(targetdata_path, "test.pt"), weights_only=True)  # test.pt
    """In pre-training: 
    train_dataset: [371055, 1, 178] from SleepEEG.
    finetune_dataset: [60, 1, 178], test_dataset: [11420, 1, 178] from Epilepsy"""

    # subset = True # if true, use a subset for debugging.
    train_dataset = Load_Dataset(train_dataset, configs, training_mode) # for self-supervised, the data are augmented here
    finetune_dataset = Load_Dataset(finetune_dataset, configs, training_mode)
    test_dataset = Load_Dataset(test_dataset, configs, training_mode)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=configs["batch_size"], shuffle=True, drop_last=configs["drop_last"], num_workers=0)
    finetune_loader = torch.utils.data.DataLoader(dataset=finetune_dataset, batch_size=configs["target_batch_size"], shuffle=True, drop_last=configs["drop_last"], num_workers=0)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=configs["target_batch_size"], shuffle=True, drop_last=False, num_workers=0)

    return train_loader, finetune_loader, test_loader