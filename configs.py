"""Experiment registry.

Each function takes the problem size ``N`` and returns a config dict
(experiment / setup). Training hyperparameters are CLI flags in train.py.
Register new configs in ``EXPERIMENTS``
at the bottom of the file to make them available via ``train.py``.
"""


def DOT_CNFM(N):
    """Complex multiplication in 2x2 real-matrix form, random split.

    Each operand a+bi is encoded as the matrix [[a, -b], [b, a]], so the
    input is a sequence of 8 tokens (two flattened matrices) and the model
    predicts all 4 entries of the product matrix. Single-head DOT.

    Training settings used in earlier runs:
        --lr 1e-3 --weight-decay 0.20 --epochs 20000 --train-frac 0.15
    """
    config = {
        "experiment": {
            "arch": "DOT",
            "dataset": "CNFM",
            "save_path": "./runs/DOT_CNFM",
            "N": N
        },
        "setup": {
            "d_model": 64,
            "n_layers": 2,
            "d_k": 128,
            "d_v": 128,
            "voc_size": 4 * N * N + 1,
            "pos_encoding": 8, # 2 matrices 2x2
            "num_outputs": 4, # 1 matrix 2x2
        }
    }

    return config


def DOT_CN(N):
    """Complex multiplication (a+bi)(c+di), random split.

    Input is the token sequence [a, b, c, d]; the model predicts the real
    and imaginary parts of the product with two output heads. Single-head
    DOT. This is the baseline the other CN* experiments vary.

    Training settings used in earlier runs:
        --lr 1e-3 --weight-decay 0.15 --epochs 15000 --train-frac 0.3
    """
    return {
        "experiment": {
            "arch": "DOT",
            "dataset": "CN",
            "save_path": "./runs/DOT_CN",
            "N": N
        },
        "setup": {
            "d_model": 64,
            "n_layers": 2,
            "d_k": 128,
            "d_v": 128,
            "voc_size": 4 * N * N + 1,
            "pos_encoding": 4,
            "num_outputs": 2,
        }
    }


def DOT_CNA(N):
    """Complex multiplication with an ordering-based split (commutativity probe).

    Same task as CN, but the model trains only on pairs with (a, b) <= (c, d)
    in lexicographic order and is evaluated on the remaining pairs.
    The split is deterministic, so ``train_frac`` is not used.

    Training settings used in earlier runs:
        --lr 1e-3 --weight-decay 0.15 --epochs 20000
    """
    return {
        "experiment": {
            "arch": "DOT",
            "dataset": "CNA",
            "save_path": "./runs/DOT_CNA",
            "N": N
        },
        "setup": {
            "d_model": 128,
            "n_layers": 2,
            "d_k": 128,
            "d_v": 128,
            "voc_size": 4 * N * N + 1,
            "pos_encoding": 4,
            "num_outputs": 2,
        }
    }


def DOT_CNC(N):
    """Complex multiplication with a sign-based split (quadrant probe).

    Same task as CN, but the model trains only on examples where both
    imaginary parts are positive (b > 0 and d > 0) and is evaluated on all
    other examples. The split is deterministic, so ``train_frac`` is not used.

    Training settings used in earlier runs:
        --lr 1e-3 --weight-decay 0.15 --epochs 20000
    """
    return {
        "experiment": {
            "arch": "DOT",
            "dataset": "CNC",
            "save_path": "./runs/DOT_CNC",
            "N": N
        },
        "setup": {
            "d_model": 128,
            "n_layers": 2,
            "d_k": 128,
            "d_v": 128,
            "voc_size": 4 * N * N + 1,
            "pos_encoding": 4,
            "num_outputs": 2,
        }
    }


def DOT_CNFMA(N):
    """Matrix-form complex multiplication with an operand-order split.

    Same encoding as CNFM. Training uses each unordered pair of complex
    numbers once in forward order; validation uses the same pairs with the
    operands swapped (excluding pairs of identical numbers). The split is
    deterministic, so ``train_frac`` is not used.

    Training settings used in earlier runs:
        --lr 1e-3 --weight-decay 0.15 --epochs 18000
    """
    return {
        "experiment": {
            "arch": "DOT",
            "dataset": "CNFMA",
            "save_path": "./runs/DOT_CNFMA",
            "N": N
        },
        "setup": {
            "d_model": 128,
            "n_layers": 1,
            "d_k": 128,
            "d_v": 128,
            "voc_size": 4 * N * N + 1,
            "pos_encoding": 8,
            "num_outputs": 4,
        }
    }


def DOT_M(p):
    """Modular addition (a + b) mod p, random split.

    The classic grokking benchmark from Power et al. (2022). Input is the
    token sequence [a, b] with a, b in [0, p); the model predicts the sum
    mod p with a single output head. Single-head DOT.

    Note: for this experiment the CLI argument ``--N`` is the modulus p,
    not the operand range used by the complex-number experiments.
    """
    return {
        "experiment": {
            "arch": "DOT",
            "dataset": "M",
            "save_path": "./runs/DOT_M",
            "N": p  # modulus
        },
        "setup": {
            "d_model": 128,
            "n_layers": 1,
            "d_k": 128,
            "d_v": 128,
            "voc_size": p,  # tokens and targets are 0..p-1
            "pos_encoding": 2,  # sequence [a, b]
            "num_outputs": 1,
        }
    }


EXPERIMENTS = {
    "DOT_M": DOT_M,
    "DOT_CNFM": DOT_CNFM,
    "DOT_CN": DOT_CN,
    "DOT_CNA": DOT_CNA,
    "DOT_CNC": DOT_CNC,
    "DOT_CNFMA": DOT_CNFMA,
}