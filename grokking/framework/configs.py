

def DOT_CNFM(N):
    config = {
        "experiment": {
            "arch": "DOT",
            "dataset": "CNFM",
            "save_path": "./runs/ex_002",
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