import torch
from torch import nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer

from src.utils import patchify


class TF_JEPA(nn.Module):
    """
    Non-contrastive cross-modal BYOL/JEPA for time-series.

    Constructor now mirrors the original code base:

        TF_JEPA(config, momentum=0.995)

    Expected `config` keys
    ----------------------
    TSlength_aligned : int   # d_model & sequence length
    embed_dim        : int   # projection size  (default 128)
    nhead            : int   # transformer heads (default 2)
    nlayers          : int   # encoder depth    (default 2)
    ff_mult          : int   # FFN expansion    (default 2)

    You can add more keys and wire them the same way.
    """

    def __init__(self, config: dict, momentum: float = 0.995):
        super().__init__()
        self.m = momentum

        # ─────── config helpers ──────────────────────────────────────────
        d_model  = config["TSlength_aligned"]
        embed    = config.get("embed_dim", 128)
        nhead    = config.get("nhead", 2)
        nlayers  = config.get("nlayers", 2)
        ff_mult  = config.get("ff_mult", 2)

        def _make_encoder():
            layer = TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=ff_mult * d_model,
                batch_first=True,
            )
            return TransformerEncoder(layer, num_layers=nlayers)

        def _make_projector():
            return nn.Sequential(
                nn.Linear(d_model, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Linear(256, embed),
            )

        # ─────── online encoders ─────────────────────────────────────────
        self.enc_time_online  = _make_encoder()
        self.enc_freq_online  = _make_encoder()
        self.proj_time_online = _make_projector()
        self.proj_freq_online = _make_projector()

        # ─────── target encoders (EMA, frozen) ───────────────────────────
        self.enc_time_target  = _make_encoder()
        self.enc_freq_target  = _make_encoder()
        self.proj_time_target = _make_projector()
        self.proj_freq_target = _make_projector()

        for p in (*self.enc_time_target.parameters(),
                  *self.enc_freq_target.parameters(),
                  *self.proj_time_target.parameters(),
                  *self.proj_freq_target.parameters()):
            p.requires_grad = False

        # ─────── predictors ──────────────────────────────────────────────
        def _make_pred():
            return nn.Sequential(
                nn.Linear(embed, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Linear(256, embed),
            )

        self.pred_time_to_freq = _make_pred()
        self.pred_freq_to_time = _make_pred()

    # ─────── EMA update helpers ──────────────────────────────────────────
    @torch.no_grad()
    def _ema(self, online: nn.Module, target: nn.Module):
        for p_o, p_t in zip(online.parameters(), target.parameters()):
            p_t.data.mul_(self.m).add_(p_o.data, alpha=1 - self.m)

    @torch.no_grad()
    def update_targets(self):
        self._ema(self.enc_time_online,  self.enc_time_target)
        self._ema(self.enc_freq_online,  self.enc_freq_target)
        self._ema(self.proj_time_online, self.proj_time_target)
        self._ema(self.proj_freq_online, self.proj_freq_target)

    # ─────── forward ─────────────────────────────────────────────────────
    def _encode(self, enc: nn.Module, proj: nn.Module, x):
        h = enc(x).mean(dim=1)          # mean-pool
        return proj(h)

    def forward(self, x_time, x_freq):
        # online
        z_t_on = self._encode(self.enc_time_online, self.proj_time_online, x_time)
        z_f_on = self._encode(self.enc_freq_online, self.proj_freq_online, x_freq)

        # targets (no-grad)
        with torch.no_grad():
            z_t_tg = self._encode(self.enc_time_target, self.proj_time_target, x_time)
            z_f_tg = self._encode(self.enc_freq_target, self.proj_freq_target, x_freq)

        # predictors
        p_t2f = self.pred_time_to_freq(z_t_on)
        p_f2t = self.pred_freq_to_time(z_f_on)

        return {
            "z_time_online":   z_t_on,
            "z_freq_online":   z_f_on,
            "z_time_target":   z_t_tg.detach(),
            "z_freq_target":   z_f_tg.detach(),
            "p_time_to_freq":  p_t2f,
            "p_freq_to_time":  p_f2t,
        }


"""Downstream classifier only used in finetuning"""
class target_classifier(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        in_dim  = 2 * cfg.get("embed_dim", 128) + cfg["input_channels"]   # auto-adapt
        hidden  = cfg.get("clf_hidden", 64)

        self.head = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LayerNorm(hidden),      # or BatchNorm1d
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, cfg["num_classes_target"])
        )

    def forward(self, z):
        z = z.view(z.size(0), -1)
        return self.head(z)


class PatchedTFC_BYOL(TF_JEPA):
    def forward(self, x_time, x_freq):
        x_time = patchify(x_time)                 # (B, N, 178)
        x_freq = patchify(x_freq)

        # call parent forward
        h_t, h_f, z_t, p_t, z_f, z_f_tgt = super().forward(x_time, x_freq)
        return h_t, h_f, z_t, p_t, z_f, z_f_tgt
