# Research

Research code on **grokking** [[1]](#references) — the delayed-generalization
phenomenon where a transformer memorizes a training set long before it
suddenly generalizes. This repo studies grokking on complex-number
multiplication tasks (and modular addition, the original grokking benchmark)
across a few custom decoder-only transformer architectures, including a
linear-attention variant inspired by neuron-astrocyte signaling.

## Structure

```
Research/
└── grokking/
    └── framework/
        ├── train.py          # CLI entry point for running experiments
        ├── configs.py        # registry of experiment configs (EXPERIMENTS)
        ├── core/
        │   ├── models.py     # architectures: DOT, MHDOT, NAT (+ GetModel factory)
        │   ├── datasets.py   # synthetic datasets + DATASET_REGISTRY / BuildDatasets
        │   ├── engine.py     # train/validate epoch loops
        │   └── utils.py      # config/checkpoint I/O, plotting, run-dir numbering
        ├── analysis/          # curated results per experiment (config + notebook + plot)
        └── runs/              # gitignored scratch output written by train.py
```

## Installation

Requires Python 3.10+ and:

```
pip install torch numpy pandas matplotlib seaborn
```

(CPU-only PyTorch is enough — the tasks here are tiny synthetic datasets.)

## Running an experiment

List what's registered:

```
python grokking/framework/train.py --list
```

Run one:

```
python grokking/framework/train.py --experiment DOT_CN --N 5
```

- `--experiment` — required, one of the names from `--list` (keys of `EXPERIMENTS` in `configs.py`)
- `--N` — required, controls the problem size (operand range `[-N, N]`, which also determines vocab size)
- `--epochs` — optional, overrides the epoch count baked into the experiment's config

Each run creates an auto-numbered directory `runs/<EXPERIMENT>/exp_XXX/` containing:
- `config.json` — the exact config used (arch, dataset, hyperparameters)
- `checkpoint_epoch_N.pt` — model + optimizer state, saved every 10 epochs
- `acc_plot.png` — train/validation accuracy curve, updated every 10 epochs

`runs/` is gitignored scratch space. When a run is worth keeping, its
`config.json`, plot, and an accompanying analysis notebook get copied into
`analysis/<EXPERIMENT>/exp_XXX/` as the permanent, curated record.

> **Note:** `acc_plot.png`/`accuracy_plot.png` titles in older `analysis/`
> folders read "Grokking Progress: Modular Addition" regardless of which
> task was actually run — that was a hardcoded plotting title, now fixed for
> new runs. Trust the folder name and `config.json` in each experiment
> directory, not the title baked into older plot images.

## Architectures (`core/models.py`)

| Name | Description |
|---|---|
| `DOT` | Decoder-only, single-head attention transformer |
| `MHDOT` | Decoder-only, multi-head attention transformer |
| `NAT` | Neuron-Astrocyte Transformer — linear-time attention via random Fourier features with an exponentially-decaying accumulated state, instead of quadratic softmax attention, inspired by [[2]](#references) |

`GetModel(config)` builds the right one from `config["experiment"]["arch"]`.
Only `DOT` experiments are currently registered in `configs.py`; `MHDOT`/`NAT`
need extra `setup` keys (`n_heads`, `m_hidden` respectively) if you register
experiments using them.

## Datasets (`core/datasets.py`)

All tasks are variants of complex multiplication `(a+bi)(c+di)`, generated
exhaustively over `a,b,c,d ∈ [-N, N]`, plus the classic modular-addition task:

| Name | Task | Train/test split |
|---|---|---|
| `M` | `(a + b) mod p` — the original grokking benchmark [[1]](#references) | random |
| `CN` | complex multiplication, scalar (re, im) output | random |
| `CNA` | same as `CN` | trained on lexicographically-ordered pairs `(a,b) ≤ (c,d)`, tested on the rest — probes whether the model generalizes commutativity |
| `CNC` | same as `CN` | trained only on `b > 0 and d > 0`, tested elsewhere — probes generalization across sign/quadrant boundaries |
| `CNFM` | complex numbers encoded as 2×2 real matrices (input/output both full matrices) | random |
| `CNFMA` | same as `CNFM` | trained on forward-order pairs, validated on the reversed-order pairs — probes order-invariance under the matrix encoding |

## Adding a new experiment

1. If it uses an existing dataset/arch: add a config-builder function to
   `configs.py` (see `DOT_CN` for the pattern) and register it in the
   `EXPERIMENTS` dict at the bottom of the file.
2. If it needs a new dataset class: add it to `core/datasets.py` and register
   it in `DATASET_REGISTRY`. If the class has a deterministic `split=`
   constructor argument (like `CNA`/`CNC`/`CNFMA`) rather than needing a
   random split, add it to `_SPLIT_AWARE` with the correct split label.
3. If it needs a new architecture: add it to `core/models.py` and wire it
   into `GetModel`.

No other file needs to change — `train.py` picks everything up through the
registries.


## References

[1] A. Power, Y. Burda, H. Edwards, I. Babuschkin, and V. Misra, "Grokking: Generalization beyond overfitting on small algorithmic datasets," arXiv:2201.02177, 2022. doi: [10.48550/arXiv.2201.02177](https://doi.org/10.48550/arXiv.2201.02177)

[2] L. Kozachkov, K. V. Kastanenka, and D. Krotov, "Building transformers from neurons and astrocytes," *Proc. Natl. Acad. Sci. U.S.A.*, vol. 120, no. 34, p. e2219150120, 2023. doi: [10.1073/pnas.2219150120](https://doi.org/10.1073/pnas.2219150120)

