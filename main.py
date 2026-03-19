#!/usr/bin/env python3
# main.py
import json, datetime
import argparse, importlib, pathlib, torch
from src.data import data_generator
from src.model import TF_JEPA, target_classifier
from src.train import pretrain_tf_jepa, finetune_tf_jepa, test_tf_jepa
from configs import (
    EXPERIMENTS, TGT_INFO, to_dict, load_config
)

# ------------------------------------------------------------------
# 1 — Parse CLI
# ------------------------------------------------------------------
def get_args():
    p = argparse.ArgumentParser(
        description="Run one or several TF-JEPA experiments sequentially."
    )
    p.add_argument("--exp", required=True,
                   help="har, ecg, fd, sleepeeg … or the word 'all' (comma-separated)")
    p.add_argument("--epochs-pre", type=int, default=5)
    p.add_argument("--epochs-ft",  type=int, default=50)
    p.add_argument("--lr",         type=float, default=3e-4)

    # load encoder instead of pre-training
    p.add_argument("--load-pretrained", default=None,
                   help="Path to a JEPA checkpoint to skip pre-training")

    # NEW — optional dataset overrides
    p.add_argument("--src",  default=None,
                   help="Override *source* dataset folder (e.g. SleepEEG)")
    p.add_argument("--tgt",  default=None,
                   help="Override *target* dataset folder (e.g. Gesture)")
    p.add_argument("--head-only", action="store_true",
                   help="During fine-tune train the classifier head only "
                        "(freeze encoder, predictors, projectors).")
    return p.parse_args()

# ------------------------------------------------------------------
# 3 — Main workflow
# ------------------------------------------------------------------
def run_experiment(exp_name, cfg_override):
    exp_cfg   = EXPERIMENTS[exp_name]
    cfg_obj   = load_config(exp_cfg["config_module"])
    cfg_dict  = to_dict(cfg_obj)

    # keep LR in sync with the CLI override
    cfg_dict["lr"] = cfg_override["lr"]

    out_dir = pathlib.Path(exp_cfg["model_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    # 0. grab CLI overrides or defaults
    src_raw = cfg_override.get("src_override") or exp_cfg["source_data"]
    tgt_raw = cfg_override.get("tgt_override") or exp_cfg["target_data"]

    # 1. helper: if no slash, prepend ./datasets/
    def normalise(d):
        return d if ("/" in d or d.startswith(".")) else f"./datasets/{d}"

    src_path, tgt_path = map(normalise, (src_raw, tgt_raw))

    # 2. auto-adjust num_classes_target / batch size
    tgt_key = pathlib.Path(tgt_path).stem          # "Gesture", "Epilepsy", …
    if tgt_key in TGT_INFO:
        num_cls, tgt_bs, loss_weights = TGT_INFO[tgt_key]
        cfg_dict["num_classes_target"] = num_cls
        cfg_dict["target_batch_size"]  = tgt_bs
        cfg_dict["loss_weights"]  = loss_weights
        print(f"✓  Updated target-set hyper-params: classes={num_cls}, batch={tgt_bs}")
    else:
        print(f"⚠  Unknown target set '{tgt_key}' – keeping num_classes_target="
              f"{cfg_dict['num_classes_target']}")

    # ---------- data ----------
    train_loader, finetune_loader, test_loader = data_generator(
        sourcedata_path=src_path,
        targetdata_path=tgt_path,
        configs=cfg_dict,
        training_mode="pre_train",
        subset=False,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = TF_JEPA(cfg_dict).to(device)

    # ==========================================================
    #  A)  Either load weights  *or*  run the pre-training loop
    # ==========================================================
    if cfg_override["load_ckpt"]:
        # 1.  Load the requested file
        ckpt_path = pathlib.Path(cfg_override["load_ckpt"]).expanduser()
        if not ckpt_path.exists():
            raise FileNotFoundError(f"--load-pretrained file not found: {ckpt_path}")
        state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state["model_state"] if "model_state" in state else state)
        jepa = model                # keep the same variable name downstream
        print(f"✓  Loaded pretrained JEPA weights from {ckpt_path}")
    else:
        # 2.  Train from scratch (previous behaviour)
        jepa = pretrain_tf_jepa(
            model, train_loader, cfg_dict,
            lr=cfg_override["lr"],
            num_epochs=cfg_override["epochs_pre"],
            alpha=0.0, beta=1.0,
        )
        torch.save(
            jepa.state_dict(),
            out_dir / f"jepa_pretrained_{cfg_override['epochs_pre']}e_"
                       f"{cfg_override['lr']}lr.ckpt",
        )

    # ---------- fine-tune ----------
    clf = target_classifier(cfg_dict).to(device)
    clf, head = finetune_tf_jepa(
        model=jepa, classifier=clf, train_loader=finetune_loader,
        config=cfg_dict, device=device, num_epochs=cfg_override["epochs_ft"],
        alpha=1.0, beta=1.0, lam=0.1, 
        freeze_backbone_epochs=3, head_only=cfg_override["head_only"],
    )

    # ---------- evaluate ----------
    mean_loss, mean_acc, mean_auc, mean_prc, f1, precision, recall, per_class = \
        test_tf_jepa(clf, head, test_loader, cfg_dict, device)

    print(
        f"[{exp_name.upper()}] "
        f"loss={mean_loss:.4f}  acc={mean_acc:.3f}  "
        f"auc={mean_auc:.3f}  prc={mean_prc:.3f}  "
        f"f1={f1:.3f}  precision={precision:.3f}  recall={recall:.3f}"
    )

    # save the usual lightweight checkpoints
    torch.save(clf.state_dict(),  out_dir / "jepa_finetuned_best.ckpt")
    torch.save(head.state_dict(), out_dir / "classifier_finetuned_best.ckpt")

    # =============================================================
    # >>> NEW:  persist run record (config + metrics)  <<<
    # =============================================================
    run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    metrics_dict = {
        "loss":       float(mean_loss),
        "accuracy":   float(mean_acc),
        "auc":        float(mean_auc),
        "prc":        float(mean_prc),
        "f1":         float(f1),
        "precision":  float(precision),
        "recall":     float(recall),
        "per_class":  per_class,
    }

    record = {
        "run_id":    run_id,
        "experiment": exp_name,
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "config":    cfg_dict,         # full hyper-param dict
        "overrides": cfg_override,     # epochs / lr from CLI
        "metrics":   metrics_dict,
    }

    # -- A) human-readable JSON summary
    summary_path = out_dir / f"{run_id}_summary.json"
    with open(summary_path, "w") as fp:
        json.dump(record, fp, indent=2)
    print(f"⇢  Saved summary  → {summary_path}")

    # -- B) monolithic checkpoint that bundles weights + record
    full_ckpt_path = out_dir / f"{run_id}_full.ckpt"
    torch.save(
        {
            "model_state": clf.state_dict(),
            "head_state":  head.state_dict(),
            "record":      record,
        },
        full_ckpt_path,
    )
    print(f"⇢  Saved weights → {full_ckpt_path}")

def main():
    args = get_args()

    exp_list = (
        list(EXPERIMENTS.keys())
        if args.exp.lower() == "all"
        else [e.strip() for e in args.exp.split(",")]
    )

    # sanity-check
    unknown = [e for e in exp_list if e not in EXPERIMENTS]
    if unknown:
        raise ValueError(f"Unknown experiment(s): {', '.join(unknown)}")

    overrides = {
        "epochs_pre": args.epochs_pre,
        "epochs_ft":  args.epochs_ft,
        "lr":         args.lr,
        "load_ckpt":  args.load_pretrained,
        "src_override": args.src,
        "tgt_override": args.tgt,
        "head_only":  args.head_only,
    }

    for e in exp_list:                 # ← sequential loop
        run_experiment(e, overrides)

# ------------------------------------------------------------------
if __name__ == "__main__":
    main()
