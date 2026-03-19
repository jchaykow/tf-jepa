# TF-JEPA

**Time‑Frequency Joint Embedding Predictive Architecture** — reference implementation of the methodology in

> *TF-JEPA: Predictive Alignment of Time–Frequency Representations Without Contrastive Pairs*

---

## Repository map

| Path / file | What it is for |
|-------------|----------------|
| **README.md** | This file — high-level overview, quick-start commands, and tips. |
| **datasets/** | All raw data. Each sub-folder contains `train.pt / val.pt / test.pt` plus optional Ray-parquet exports. |
| &nbsp;&nbsp;  └─ **ECG** • **EMG** • **Epilepsy** • **FD-A** • **FD-B** • **Gesture** • **HAR** • **SleepEEG** | Eight benchmark datasets used in the paper / experiments. |
| **db_notebooks/** | Databricks notebooks that show interactive training & evaluation; useful for exploratory work, optional for CLI runs. |
| &nbsp;&nbsp;  └─ `*.py` | One notebook per domain shift (e.g. *SleepEEG → Epilepsy*). |
| **main.py** | Single-process pipeline: pre-train **→** fine-tune on CPU or one GPU. |
| **main_ray.py** | Ray-based launcher that runs the same pipeline on a local or remote Ray cluster (multi-GPU / multi-node). |
| **models/** | Output artefacts — pretrained encoders, fine-tuned heads, JSON run summaries, monolithic checkpoints. Auto-created by `main.py` / `main_ray.py`. |
| &nbsp;&nbsp;  └─ **ecg**, **epilepsy**, **fd**, **har**, **sleepeeg** | One sub-folder per experiment key. |
| **pyproject.toml** | Poetry/UV build file — declares package name (`tf-jepa`) and Python / dependency versions. |
| **scripts/** | Small utilities you run from the CLI. |
| &nbsp;&nbsp;  ├─ `download_datasets.sh` | Bash helper that fetches and unpacks all datasets into `./datasets/`. |
| &nbsp;&nbsp;  └─ `export_to_ray_datasets.py` | Converts every `<dataset>/*.pt` split to Ray-AIR parquet (needed once before launching `main_ray.py`). |
| **src/** | Importable Python package (`tf_jepa`). All library code lives here. |
| &nbsp;&nbsp;  ├─ `__init__.py` | Makes `src/` a package so you can `pip install -e .`. |
| &nbsp;&nbsp;  ├─ **config_files/** | Pure-Python `Config` classes — hyper-params for each dataset. |
| &nbsp;&nbsp;  ├─ `configs.py` | Global registry (`EXPERIMENTS`, `TGT_INFO`, helpers to flatten configs). |
| &nbsp;&nbsp;  ├─ `data.py` | Local PyTorch `Dataset` / `DataLoader` logic used by `main.py`. |
| &nbsp;&nbsp;  ├─ **distributed/** | Code that only Ray workers use. |
| &nbsp;&nbsp;  │   ├─ `pretrain_loop.py` | Worker loop for TF-JEPA pre-training. |
| &nbsp;&nbsp;  │   ├─ `finetune_loop.py` | Worker loop for supervised fine-tuning. |
| &nbsp;&nbsp;  │   └─ `ray_datasets.py`  | Helper to build or load Ray-AIR datasets from the raw `.pt` files. |
| &nbsp;&nbsp;  ├─ `loss.py`  | NT-Xent variant and any other custom loss functions. |
| &nbsp;&nbsp;  ├─ `model.py` | Implementation of the **TF_JEPA** backbone and the `target_classifier` head. |
| &nbsp;&nbsp;  ├─ `train.py` | Local (non-Ray) training utilities: `pretrain_tf_jepa`, `finetune_tf_jepa`, `test_tf_jepa`. |
| &nbsp;&nbsp;  └─ `utils.py` | Augmentations and misc helpers shared by local & distributed code. |

---

## Installation

Python ≥3.12 and torch ≥2.7 and ray[train]>=2.45.0 recommended.

---

## Quick Start

### 1 – Fetch the data

Use `download_datasets.sh` to retrieve data in .pt format.

| Scenario # | Phase        | Dataset    | # Samples          | # Channels | # Classes | Length | Freq&nbsp;(Hz) |
|-----------:|--------------|------------|--------------------|-----------:|----------:|-------:|--------------:|
| **1** | Pre-training | SleepEEG | 371 ,055 | 1 | 5 | 200 | 100 |
|        | Fine-tuning  | Epilepsy | 60 / 20 / 11 ,420 | 1 | 2 | 178 | 174 |
| **2** | Pre-training | FD-A | 8 ,184 | 1 | 3 | 5 ,120 | 64 K |
|        | Fine-tuning  | FD-B | 60 / 21 / 13 ,559 | 1 | 3 | 5 ,120 | 64 K |
| **3** | Pre-training | HAR | 10 ,299 | 9 | 6 | 128 | 50 |
|        | Fine-tuning  | Gesture | 320 / 120 / 120 | 3 | 8 | 315 | 100 |
| **4** | Pre-training | ECG | 43 ,673 | 1 | 4 | 1 ,500 | 300 |
|        | Fine-tuning  | EMG | 122 / 41 / 41 | 1 | 3 | 1 ,500 | 4 ,000 |


### 2 – Pre‑train + Finetune TF‑JEPA

To run an experiement:

```bash
# ──────────────────────────────────────────────
# 1.  Train the default HAR → Gesture pipeline
# ──────────────────────────────────────────────
uv run python main.py --exp har


# ──────────────────────────────────────────────
# 2.  Same run but longer schedules and lower LR
# ──────────────────────────────────────────────
uv run python main.py --exp har \
                      --epochs-pre 20 \
                      --epochs-ft  40 \
                      --lr 1e-4


# ──────────────────────────────────────────────
# 3.  Cross-domain: SleepEEG ➜ Gesture
#    (override only the target dataset)
# ──────────────────────────────────────────────
uv run python main.py --exp sleepeeg --tgt Gesture


# ──────────────────────────────────────────────
# 4.  Cross-domain: SleepEEG ➜ EMG
#    (override both source and target)
# ──────────────────────────────────────────────
uv run python main.py --exp sleepeeg \
                      --src SleepEEG \
                      --tgt EMG


# ──────────────────────────────────────────────
# 5.  Fine-tune using an existing JEPA encoder
#    (skip the pre-training loop completely)
# ──────────────────────────────────────────────
uv run python main.py --exp ecg \
                      --load-pretrained ./models/ecg/jepa_pretrained_5e_0.0003lr.ckpt


# ──────────────────────────────────────────────
# 6.  Sweep two datasets sequentially
# ──────────────────────────────────────────────
uv run python main.py --exp har,fd


# ──────────────────────────────────────────────
# 7.  Sweep *all* experiments with a shared
#    pretrained encoder, one after another
# ──────────────────────────────────────────────
uv run python main.py --exp all \
                      --load-pretrained ./models/sleepeeg/global_jepa.ckpt


# ──────────────────────────────────────────────
# 8.  Quick linear-probe: freeze backbone,
#    fine-tune head for just 5 epochs
# ──────────────────────────────────────────────
uv run python main.py --exp ecg \
                      --epochs-pre 0 \
                      --epochs-ft  5
```

| Keyword (`--src` / `--tgt`) | Folder it resolves to | Typical role |
|------------------------------|-----------------------|--------------|
| `HAR` | `./datasets/HAR` | sensor **source** (Human-Activity) |
| `Gesture` | `./datasets/Gesture` | sensor **target** (phone gestures) |
| `ECG` | `./datasets/ECG` | biosignal **source** |
| `EMG` | `./datasets/EMG` | biosignal **target** |
| `FD-A` | `./datasets/FD-A` | bearing-fault **source** |
| `FD-B` | `./datasets/FD-B` | bearing-fault **target** |
| `SleepEEG` | `./datasets/SleepEEG` | EEG **source** |
| `Epilepsy` | `./datasets/Epilepsy` | EEG **target** |

## Outputs

| Artifact | File name pattern | Where it lives | What’s inside / why you need it |
|----------|------------------|----------------|---------------------------------|
| Pre-trained encoder | `jepa_pretrained_<E>e_<LR>lr.ckpt` | `./models/<exp>/` | The JEPA backbone after the pre-training stage (or re-saved when you loaded a checkpoint). Handy if you want to reuse the encoder elsewhere without the classifier. |
| Best fine-tuned encoder | `jepa_finetuned_best.ckpt` | same folder | State-dict of the encoder **after** fine-tuning. |
| Best classifier head | `classifier_finetuned_best.ckpt` | same folder | Linear-/MLP-head weights trained on the target dataset. |
| Run summary (JSON) | `<RUN_ID>_summary.json` | same folder | Human-readable record of the entire run:<br>• hyper-parameters (flattened `cfg_dict`)<br>• CLI overrides (epochs, LR, dataset overrides, ckpt path)<br>• final metrics on the test set (`loss`, `accuracy`, `auc`, `prc`, `f1`, `precision`, `recall`)<br>• time-stamp & unique `run_id`. |
| Monolithic checkpoint | `<RUN_ID>_full.ckpt` | same folder | `torch.save({...})` bundle that contains:<br>```python<br>{<br>  "model_state": <encoder-state>,<br>  "head_state":  <classifier-state>,<br>  "record":      <same JSON dict><br>}<br>```<br>Load once and you have weights **plus** the exact config & metrics that produced them. |

## Databricks Notebooks

- **db\_notebooks/** contain ready‑to‑run notebooks that mount your workspace, use `ray_datasets.py` to convert from .pt data to ray datasets.

## Ray Data & Ray Training

### Flag reference

| Flag | Default | Description |
|------|---------|-------------|
| `--exp`            | _(required)_ | Experiment key `har ecg fd sleepeeg` or comma-list / `all`. |
| `--src`, `--tgt`   | registry default | Override source / target dataset folders or keywords. |
| `--num-workers`    | `4` | Workers per **pre-train** trainer. Fine-tune always uses `1`. |
| `--use-gpu`        | *off* | If present, each worker reserves **one GPU**. Omit for CPU-only. |
| `--epochs-pre`     | `20` | Pre-train epochs. |
| `--epochs-ft`      | `40` | Fine-tune epochs. |
| `--lr`             | `3e-4` | Learning rate for *both* stages. |
| `--ckpt-freq`      | `5` | Save a Ray `Checkpoint` every *N* pre-train epochs. |
| `--load-pretrained`| *None* | Path to a `.ckpt / .pth / .pt` file → skips the pre-train stage. |

### Copy-paste examples

```bash
# 1 ▶ default (CPU) SleepEEG → Epilepsy, 4 workers
uv run python main_ray.py \
    --exp sleepeeg \
    --num-workers 4

# 2 ▶ same but on every worker’s GPU
uv run python main_ray.py \
    --exp sleepeeg \
    --num-workers 4 \
    --use-gpu

# 3 ▶ cross-domain: SleepEEG ➜ Gesture
uv run python main_ray.py \
    --exp sleepeeg \
    --tgt Gesture \
    --num-workers 4

# 4 ▶ skip pre-train – start from an existing encoder
uv run python main_ray.py \
    --exp sleepeeg \
    --load-pretrained ./models/sleepeeg/jepa_pretrained_5e_0.0003lr.ckpt \
    --num-workers 1             # ← only need fine-tune workers

# 5 ▶ longer fine-tune (150 epochs)
uv run python main_ray.py \
    --exp sleepeeg \
    --epochs-ft 150 \
    --num-workers 2

# 6 ▶ pre-train checkpoints every 2 epochs (FD-A → FD-B)
uv run python main_ray.py \
    --exp fd \
    --ckpt-freq 2 \
    --num-workers 4
```

## Troubleshooting

| Symptom             | Cause                      | Fix                                               |
| ------------------- | -------------------------- | ------------------------------------------------- |
| Loss collapses to 0 | Predictor collapse         | Lower momentum (0.98) or add predictor hidden 512 |
| CUDA OOM            | Long sequence or big batch | Reduce `batch_size` or set `max_len` in config    |
|                     |                            |                                                   |

## Acknowledgements

Built upon ideas from BYOL, JEPA, TF-C, and the broader self‑supervised learning community.

## Baseline Notes

All datasets use the same format of .pt files that are downloaded from ./scripts/download_datasets.sh and any baselines that require modified data formats have scripts to do so.

### Normwear Baseline

Normwear baseline requires download of the pretrained foundation model found [here](https://github.com/Mobile-Sensing-and-UbiComp-Laboratory/NormWear/releases/tag/v1.0.0-alpha)

The same .pt files can be placed in ./normwear/data/ the same as every other method, then you can run each convert_<dataset>_data.py file to generate the correct data structure for finetuning.

### TF-C and TS-TCC Baselines

TF-C and TS-TCC have a very similar directory structure as TF-JEPA for dataset loading.
