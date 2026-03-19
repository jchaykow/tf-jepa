import torch, numpy as np, os
from ray.air import session
from ray.train import Checkpoint
import pickle, logging, sys, pathlib
from torch.nn import CrossEntropyLoss
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
)

from model import TF_JEPA, target_classifier


def finetune_loop_per_worker(cfg):
    """
    A Ray-AIR worker loop that fine-tunes a pretrained TF-JEPA encoder +
    classifier on the shard it receives.  Expects the following keys in `cfg`:

    ─ cfg["cfg_dict"]        : full model hyper-param dict (flattened)
    ─ cfg["seed"]            : RNG seed
    ─ cfg["lr"]              : learning-rate
    ─ cfg["num_epochs"]      : # fine-tuning epochs
    ─ cfg["batch_size"]      : batch size per worker
    ─ cfg.get("pretrain_ckpt"): optional Ray Checkpoint with encoder weights
    """
    # -----------------------------------------------------------
    # 0 ─ Logging boilerplate
    # -----------------------------------------------------------
    logger = logging.getLogger("ray_finetune_worker")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        hdlr = logging.StreamHandler(sys.stdout)
        hdlr.setFormatter(logging.Formatter("%(asctime)s %(levelname)s: %(message)s"))
        logger.addHandler(hdlr)

    # -----------------------------------------------------------
    # 1 ─ Reproducibility
    # -----------------------------------------------------------
    seed = cfg.get("seed", 0)
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -----------------------------------------------------------
    # 2 ─ Model & classifier
    # -----------------------------------------------------------
    cfg_dict = cfg["cfg_dict"]
    encoder  = TF_JEPA(cfg_dict).to(device)
    clf      = target_classifier(cfg_dict).to(device)

    # (optional) load encoder state from Ray Checkpoint
    state_dict = None
    ckpt_src   = cfg.get("pretrain_ckpt")

    if ckpt_src:
        # ─────────────────── Ray Checkpoint object ───────────────────
        if isinstance(ckpt_src, Checkpoint):
            with ckpt_src.as_directory() as d:
                d = pathlib.Path(d)
                file = d / "model_state.pt"
                if not file.exists():
                    # fall back to the first *.pt / *.pth / *.pkl in the dir
                    matches = list(d.glob("*.pt")) + list(d.glob("*.pth")) + list(d.glob("*.pkl"))
                    if not matches:
                        logging.warning(f"No weight file found in Ray checkpoint dir {d}")
                    else:
                        file = matches[0]
                if file.exists():
                    state_dict = torch.load(file, map_location="cpu")

        # ─────────────────── Plain file path ─────────────────────────
        else:
            path = os.path.expanduser(str(ckpt_src))
            if os.path.exists(path):
                # accepts both dict-with-key or bare state-dict
                loaded = torch.load(path, map_location="cpu")
                state_dict = loaded.get("model_state", loaded) if isinstance(loaded, dict) else loaded
            else:
                logging.warning(f"Checkpoint file {path} not found")

    # ───────────── finally load if we have a state-dict ─────────────
    if state_dict:
        encoder.load_state_dict(state_dict, strict=False)
        logging.info("✓  Pretrained encoder weights loaded.")
    else:
        logging.info("⚠️  Proceeding without pretrained weights.")

    # only fine-tune encoder + classifier
    params = list(encoder.parameters()) + list(clf.parameters())
    optim  = torch.optim.Adam(params, lr=cfg["lr"], betas=(0.9, 0.99), weight_decay=3e-4)

    # -----------------------------------------------------------
    # 3 ─ Ray datasets
    # -----------------------------------------------------------
    ds_train = session.get_dataset_shard("finetune")
    ds_test  = session.get_dataset_shard("test")
    bs       = cfg["batch_size"]
    epochs   = cfg["num_epochs"]
    criterion = CrossEntropyLoss()

    # -----------------------------------------------------------
    # 4 ─ Training loop
    # -----------------------------------------------------------
    for ep in range(1, epochs + 1):
        encoder.train(); clf.train()
        running_loss, n = 0.0, 0

        for batch in ds_train.iter_batches(batch_size=bs,
                                        batch_format="numpy", drop_last=False):
            # --------------- prepare data -----------------------
            x = torch.tensor(batch["x_data"], dtype=torch.float32, device=device)
            y = torch.tensor(batch["y_data"], dtype=torch.long,   device=device)
            x_f = torch.abs(torch.fft.fft(x, dim=-1))

            # --------------- forward & loss --------------------
            optim.zero_grad()
            # TF_JEPA returns (h_t, h_f, z_t, p_t, z_f, zf_tgt)
            _, _, z_t, _, z_f, _ = encoder(x, x_f)               # ← pass both views
            features = torch.cat([z_t, z_f], dim=1)
            logits   = clf(features)
            loss   = criterion(logits, y)
            loss.backward(); optim.step()

            running_loss += loss.item(); n += 1

        avg_loss = running_loss / max(n, 1)

        # -------------------------------------------------------
        # 5 ─ Evaluation
        # -------------------------------------------------------
        encoder.eval(); clf.eval()
        preds, labels, logits_all = [], [], []

        with torch.no_grad():
            for batch in ds_test.iter_batches(batch_size=bs,
                                            batch_format="numpy", drop_last=False):
                x = torch.tensor(batch["x_data"], dtype=torch.float32, device=device)
                y = torch.tensor(batch["y_data"], dtype=torch.long,   device=device)
                x_f = torch.abs(torch.fft.fft(x, dim=-1))

                _, _, z_t, _, z_f, _ = encoder(x, x_f)
                features = torch.cat([z_t, z_f], dim=1)
                logits   = clf(features)

                preds.append(logits.argmax(1).cpu().numpy())
                labels.append(y.cpu().numpy())
                logits_all.append(logits.cpu().numpy())

        preds   = np.concatenate(preds)
        labels  = np.concatenate(labels)
        logits_ = np.concatenate(logits_all)

        acc  = accuracy_score(labels, preds)
        prec = precision_score(labels, preds, average="macro")
        rec  = recall_score(labels, preds, average="macro")
        f1   = f1_score(labels, preds, average="macro")

        k = cfg_dict["num_classes_target"]
        oh = np.zeros((labels.size, k)); oh[np.arange(labels.size), labels] = 1
        try:
            auc = roc_auc_score(oh, logits_, average="macro", multi_class="ovr")
        except ValueError:
            auc = 0.0
        prc = average_precision_score(oh, logits_, average="macro")

        logger.info(f"[Epoch {ep}/{epochs}] "
                    f"loss={avg_loss:.4f} acc={acc:.3f} f1={f1:.3f} "
                    f"auc={auc:.3f} prc={prc:.3f}")

        session.report({
            "epoch": ep,
            "finetune_loss": avg_loss,
            "acc": acc,
            "precision": prec,
            "recall": rec,
            "f1_score": f1,
            "auroc": auc,
            "auprc": prc,
        })

