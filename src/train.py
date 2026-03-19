import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score
)
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
from src.loss import NTXentLoss_poly, cosine_byol
from src.model import TF_JEPA


def pretrain_tf_jepa(
    model,
    train_loader,
    config,
    lr=1e-3,
    num_epochs=10,
    alpha=1.0,
    beta=1.0            # ← ditto
):
    """Pure BYOL/JEPA self‑supervised pre‑train (no negatives)."""

    symmetric = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).train()

    # optimiser (all trainable params)
    opt = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        betas=(0.9, 0.99),
        weight_decay=3e-4,
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=num_epochs)

    for epoch in range(1, num_epochs + 1):
        losses = []
        for x_t, _, _, x_f, _ in train_loader:  # loader must yield time & freq tensors
            x_t = x_t.float().to(device)
            x_f = x_f.float().to(device)

            opt.zero_grad()

            # two views: original + random augmentation (shuffling handled outside)
            out1 = model(x_t, x_f)
            out2 = model(x_t.flip(1), x_f.flip(1))  # trivial augmentation example

            # BYOL losses -------------------------------------------------
            loss_main = cosine_byol(out1["p_time_to_freq"], out1["z_freq_target"]) + \
                        cosine_byol(out2["p_time_to_freq"], out2["z_freq_target"])

            if symmetric:
                loss_main += cosine_byol(out1["p_freq_to_time"], out1["z_time_target"]) + \
                              cosine_byol(out2["p_freq_to_time"], out2["z_time_target"])
                loss_main *= 0.5  # average the two directions

            loss = loss_main
            loss.backward()
            opt.step()
            model.update_targets()

            losses.append(loss.item())

        mean_loss = float(np.mean(losses))
        print(f"Epoch {epoch:02d}/{num_epochs}  loss={mean_loss:.4f}")
        sched.step()

    return model

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score
import numpy as np


def finetune_tf_jepa(
    model: TF_JEPA,
    classifier: nn.Module,
    train_loader,
    config,
    device,
    num_epochs: int = 20,
    alpha: float = 0.5,          # weight on TF-C loss
    beta: float  = 0.5,          # weight on JEPA loss
    lam: float   = 0.1,          # intra-view weighting inside TF-C
    freeze_backbone_epochs: int = 3,
    head_only: bool = False
):
    """
    • freezes backbone for the first few epochs (only predictors + head learn)
    • then unfreezes encoders/projectors at a lower LR
    • uses NT-Xent for TF-C  +  cosine BYOL for JEPA
    """

    model.to(device)
    classifier.to(device)

    # ─────────────────────────────────────────────
    # 1 ▸ freeze backbone, keep predictors + head
    # ─────────────────────────────────────────────
    for p in model.parameters():
        p.requires_grad = False

    for p in (
        *model.pred_time_to_freq.parameters(),
        *model.pred_freq_to_time.parameters(),
    ):
        p.requires_grad = True

    for p in classifier.parameters():
        p.requires_grad = True

    # optimiser
    optimizer = torch.optim.AdamW(
        [ {"params": p} for p in
          list(classifier.parameters()) +
          ([] if head_only else
           list(model.pred_time_to_freq.parameters()) +
           list(model.pred_freq_to_time.parameters()))
        if p.requires_grad ],
        lr=1e-4, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    # losses
    counts = torch.tensor(config["loss_weights"], dtype=torch.float, device=device)
    weights = 1.0 / counts.sqrt()           # √-inverse instead of full inverse
    weights /= weights.sum()                # normalise
    criterion_cls = nn.CrossEntropyLoss(weight=weights)

    def jepa_loss(p, z):
        z = z.detach()
        p = F.normalize(p, dim=-1)
        z = F.normalize(z, dim=-1)
        return 2 - 2 * (p * z).sum(-1).mean()      # BYOL-style cosine

    ntxent_loss_fn = NTXentLoss_poly(
        device=device,
        batch_size=config["target_batch_size"],
        temperature=config["Context_Cont"]["temperature"],
        use_cosine_similarity=config["Context_Cont"]["use_cosine_similarity"],
    )

    for epoch in range(num_epochs):

        if (not head_only) and epoch == freeze_backbone_epochs:
            new_group = []
            for name, p in model.named_parameters():
                if name.startswith(("enc_time_online", "enc_freq_online",
                                    "proj_time_online", "proj_freq_online")):
                    p.requires_grad = True
                    new_group.append(p)
            if new_group:
                optimizer.add_param_group({"params": new_group, "lr": 1e-5})

        model.train(); classifier.train()
        cum_loss = cum_acc = cum_auc = cum_ap = 0.0

        for (x_t, y, x_t_aug, x_f, x_f_aug) in train_loader:
            x_t, x_f       = x_t.float().to(device),     x_f.float().to(device)
            x_t_aug, x_f_aug = x_t_aug.float().to(device), x_f_aug.float().to(device)
            y   = y.long().to(device)

            optimizer.zero_grad()

            # forward ─ original & augmented view
            out      = model(x_t,      x_f)
            out_aug  = model(x_t_aug,  x_f_aug)

            # ─────────────────── TF-C (contrastive) ───────────────────
            loss_t  = ntxent_loss_fn(out["z_time_online"],
                                     out_aug["z_time_online"])
            loss_f  = ntxent_loss_fn(out["z_freq_online"],
                                     out_aug["z_freq_online"])
            loss_tf = ntxent_loss_fn(out["z_time_online"],
                                     out["z_freq_online"])
            tfc_loss = lam * (loss_t + loss_f) + loss_tf

            # ─────────────────── JEPA / BYOL (non-contrastive) ───────────────────
            jepa_main = jepa_loss(out["p_time_to_freq"], out["z_freq_target"]) + \
                        jepa_loss(out["p_freq_to_time"], out["z_time_target"])

            jepa_aug  = jepa_loss(out_aug["p_time_to_freq"], out_aug["z_freq_target"]) + \
                        jepa_loss(out_aug["p_freq_to_time"], out_aug["z_time_target"])

            jepa_loss_val = 0.5 * (jepa_main + jepa_aug)

            raw_stat = x_t.mean(dim=-1)          # shape (B, C) == (B,1)
            raw_stat = raw_stat.view(raw_stat.size(0), -1)
            # ─────────────────── classification head ───────────────────
            feats   = torch.cat([out["z_time_online"].detach(),
                                 out["z_freq_online"].detach(),
                                 raw_stat], dim=1)
            logits  = classifier(feats)
            cls_loss = criterion_cls(logits, y)

            # total
            if head_only:
                total_loss = cls_loss
            else:
                total_loss = cls_loss + alpha * tfc_loss + beta * jepa_loss_val
            total_loss.backward()
            optimizer.step()
            if not head_only:
                model.update_targets()

            # ─────────── metrics ───────────
            with torch.no_grad():
                prob = logits.softmax(dim=1).cpu()
                pred = prob.argmax(dim=1)
                cum_acc += (pred == y.cpu()).float().mean().item()

                y_onehot = F.one_hot(y.cpu(), config["num_classes_target"]).float()
                try:
                    cum_auc += roc_auc_score(y_onehot, prob,
                                             average="macro", multi_class="ovr")
                except ValueError:
                    cum_auc += 0.0
                cum_ap  += average_precision_score(y_onehot, prob,
                                                   average="macro")
                cum_loss += total_loss.item()

        scheduler.step()
        n = len(train_loader)
        print(f"[Ep {epoch+1:02}/{num_epochs}] "
              f"loss={cum_loss/n:.4f}  "
              f"acc={100*cum_acc/n:.1f}%  "
              f"auc={cum_auc/n:.3f}  "
              f"ap={cum_ap/n:.3f}")

    return model, classifier


def test_tf_jepa(model, classifier, test_loader, config, device):
    """
    Evaluates the frozen backbone + classifier.
    Uses the *projected* embeddings returned by the new TF-JEPA model.
    """
    model.eval()
    classifier.eval()

    ce = nn.CrossEntropyLoss()

    loss_hist, acc_hist, auc_hist, prc_hist = [], [], [], []
    all_preds, all_labels = [], []

    with torch.no_grad():
        for x_t, y, _, x_f, _ in test_loader:           # loader: time, label, aug, freq, augF
            x_t, x_f = x_t.float().to(device), x_f.float().to(device)
            y        = y.long().to(device)

            out   = model(x_t, x_f)                     # dict output
            raw_stat = x_t.mean(dim=-1)          # shape (B, C) == (B,1)
            raw_stat = raw_stat.view(raw_stat.size(0), -1)
            feats = torch.cat([out["z_time_online"],
                                out["z_freq_online"],
                                raw_stat], dim=1)

            logits = classifier(feats)

            # ───────── losses & metrics ─────────
            loss_hist.append(ce(logits, y).item())

            prob = logits.softmax(dim=1).cpu()
            pred = prob.argmax(dim=1)
            acc_hist.append((pred == y.cpu()).float().mean().item())

            y_onehot = F.one_hot(y.cpu(), config["num_classes_target"]).float()
            try:
                auc = roc_auc_score(y_onehot, prob, average="macro", multi_class="ovr")
            except ValueError:
                auc = 0.0
            prc = average_precision_score(y_onehot, prob, average="macro")

            auc_hist.append(auc)
            prc_hist.append(prc)

            all_preds.append(pred.numpy())
            all_labels.append(y.cpu().numpy())

    # aggregate
    mean_loss = float(np.mean(loss_hist))
    mean_acc  = float(np.mean(acc_hist))
    mean_auc  = float(np.mean(auc_hist))
    mean_ap   = float(np.mean(prc_hist))

    all_preds  = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    precision = precision_score(all_labels, all_preds, average="macro")
    recall    = recall_score(all_labels,  all_preds, average="macro")
    f1        = f1_score(all_labels,     all_preds, average="macro")
    precision_c, recall_c, f1_c, _ = precision_recall_fscore_support(
        all_labels, all_preds, average=None, labels=np.arange(config["num_classes_target"])
    )
    cmatrix = confusion_matrix(all_labels, all_preds, labels=np.arange(config["num_classes_target"]))
    # convert to python scalars for JSON-serialisation
    per_class = {
        "precision": precision_c.tolist(),
        "recall":    recall_c.tolist(),
        "f1":        f1_c.tolist(),
        "confusion": cmatrix.tolist(),
    }

    print(f"[Test] "
          f"Loss={mean_loss:.4f}  "
          f"Acc={mean_acc*100:.2f}%  "
          f"Precision={precision*100:.2f}%  "
          f"Recall={recall*100:.2f}%  "
          f"F1={f1*100:.2f}%  "
          f"AUROC={mean_auc*100:.2f}%  "
          f"AUPRC={mean_ap*100:.2f}%")

    return mean_loss, mean_acc, mean_auc, mean_ap, f1, precision, recall, per_class
