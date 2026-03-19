# src/train_ray.py
import torch, numpy as np
from ray.air import session
from ray.train import Checkpoint
import pickle, pathlib

from model import TF_JEPA
from loss import NTXentLoss_poly
from utils import DataTransform_TD, DataTransform_FD


def pretrain_loop_per_worker(cfg):
    """
    Ray worker loop that pre-trains TF-JEPA on its dataset shard.

    cfg fields expected
    -------------------
    • cfg["cfg_dict"]      : flattened hyper-param dict
    • cfg["seed"]          : RNG seed
    • cfg["lr"]            : learning rate
    • cfg["batch_size"]    : batch size per worker
    • cfg["num_epochs"]    : # epochs
    • cfg["ckpt_freq"]     : save checkpoint every N epochs (optional)
    • cfg["momentum"]      : EMA momentum for target encoder (default 0.99)
    """

    # ─────────────────────────────────────────────────────────────
    # 0  Reproducibility & device
    # ─────────────────────────────────────────────────────────────
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ─────────────────────────────────────────────────────────────
    # 1  Model & optimiser
    # ─────────────────────────────────────────────────────────────
    model_cfg = cfg["cfg_dict"]
    model = TF_JEPA(model_cfg, momentum=cfg.get("momentum", 0.99)).to(device)

    optim = torch.optim.Adam(
        list(model.online_time_encoder.parameters()) +
        list(model.online_predictor.parameters()),
        lr=cfg["lr"], betas=(0.9, 0.99), weight_decay=3e-4
    )

    ntxent = NTXentLoss_poly(
        device=device,
        batch_size=cfg["batch_size"],
        temperature=model_cfg["Context_Cont"]["temperature"],
        use_cosine_similarity=model_cfg["Context_Cont"]["use_cosine_similarity"],
    )

    # ─────────────────────────────────────────────────────────────
    # 2  Ray Dataset shards
    # ─────────────────────────────────────────────────────────────
    ds = session.get_dataset_shard("train")
    bs, epochs = cfg["batch_size"], cfg["num_epochs"]

    for ep in range(1, epochs + 1):
        model.train()
        running_loss, n = 0.0, 0

        for batch in ds.iter_batches(batch_size=bs, batch_format="numpy", drop_last=True):
            # --- prepare data --------------------------------------------------
            x = torch.as_tensor(batch["x_data"], dtype=torch.float32, device=device)
            x_aug_t = DataTransform_TD(x, model_cfg)

            x_f     = torch.abs(torch.fft.fft(x,      dim=-1))
            x_aug_f = torch.abs(torch.fft.fft(x_aug_t, dim=-1))

            # --- forward / loss ----------------------------------------------
            optim.zero_grad()

            p_t, z_f = model(x, x_f)               # online-T  vs target-F
            loss_1   = ntxent(p_t, z_f)

            p_f, z_t = model(x_f, x)               # swap roles
            loss_2   = ntxent(p_f, z_t)

            loss = 0.5 * (loss_1 + loss_2)
            loss.backward(); optim.step()
            model.update_momentum()

            running_loss += loss.item(); n += 1

        avg = running_loss / max(n, 1)
        print(f"[Worker {session.get_world_rank()}] Epoch {ep}/{epochs}  loss={avg:.4f}")

        # ─ checkpoint every cfg["ckpt_freq"] epochs ─
        if cfg.get("ckpt_freq") and ep % cfg["ckpt_freq"] == 0:
            # Put every file you want into a directory Ray can archive
            tmp_dir = pathlib.Path("/tmp/ray_jepa_ckpt") / f"epoch_{ep}"
            tmp_dir.mkdir(parents=True, exist_ok=True)

            # Save *only* the state-dict you need; optimizer is optional
            torch.save({"model_state": model.state_dict()}, tmp_dir / "model_state.pt")

            # Wrap the directory into a Ray Checkpoint object
            session.report(
                {"epoch": ep, "loss": avg},
                checkpoint=Checkpoint.from_directory(tmp_dir)
            )
        else:
            session.report({"epoch": ep, "loss": avg})
