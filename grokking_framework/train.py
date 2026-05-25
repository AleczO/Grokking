import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import json
import argparse

from core.models import GetModel
from core.utils import SaveConfig, SaveModel, LoadModel, PlotModel, DivideDataset
from core.datasets import complexNumbers
from core.engine import TrainEpoch, Validate


def train():
    config = {
        "experiment": {
            "arch": "NAT",
            "dataset": "complex_numbers",
            "save_path": "./runs/ex_001",
            "N": 4
        },
        "setup": {
            "d_model": 64,
            "n_layers": 1,
            "m_hidden": 128,
            "pos_encoding": 4,
            "voc_size": 65
        },
        "training": {
            "lr": 1e-3,
            "weight_decay": 0.10,
            "epochs": 20000
        }
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    dataset = complexNumbers(config["experiment"]["N"])

    train_ds, val_ds = DivideDataset(dataset, config["training"]["train_frac"])
    train_loader = DataLoader(train_ds, batch_size=config["training"]["batch_size"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config["training"]["batch_size"], shuffle=False)

    model = GetModel(config)
    model.to(device)

    optimizer = optim.AdamW(
        model.parameters(), 
        lr=config["training"]["lr"], 
        weight_decay=config["training"]["weight_decay"]
    )

    print("Starting training...")

    for e in range(config["training"]["epochs"]):
        TrainEpoch(model, train_loader, optimizer, device)
        Validate(model, val_loader, device)


if __name__ == "__main__":
    train()