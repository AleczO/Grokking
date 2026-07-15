

def DOT_CNFM(N):
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
        },
        "training": {
            "lr": 1e-3,
            "weight_decay": 0.20,
            "epochs": 20000,
            "train_frac": 0.15,
            "batch_size": 256
        }
    }

    return config


def DOT_CN(N):
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
        },
        "training": {
            "lr": 1e-3,
            "weight_decay": 0.15,
            "epochs": 15000,
            "train_frac": 0.3,
            "batch_size": 256
        }
    }


def DOT_CNA(N):
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
        },
        "training": {
            "lr": 1e-3,
            "weight_decay": 0.15,
            "epochs": 20000,
            "train_frac": 0.4,
            "batch_size": 256
        }
    }


def DOT_CNC(N):
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
        },
        "training": {
            "lr": 1e-3,
            "weight_decay": 0.15,
            "epochs": 20000,
            "train_frac": 0.4,
            "batch_size": 256
        }
    }


def DOT_CNFMA(N):
    return {
        "experiment": {
            "arch": "DOT",
            "dataset": "CNFMA",
            "save_path": "./runs/DOT_CNFMA",
            "N": N
        },
        "setup": {
            "d_model": 128,
            "n_layers": 2,
            "d_k": 128,
            "d_v": 128,
            "voc_size": 4 * N * N + 1,
            "pos_encoding": 8,
            "num_outputs": 4,
        },
        "training": {
            "lr": 1e-3,
            "weight_decay": 0.15,
            "epochs": 18000,
            "train_frac": 0.4,
            "batch_size": 256
        }
    }


EXPERIMENTS = {
    "DOT_CNFM": DOT_CNFM,
    "DOT_CN": DOT_CN,
    "DOT_CNA": DOT_CNA,
    "DOT_CNC": DOT_CNC,
    "DOT_CNFMA": DOT_CNFMA,
}
