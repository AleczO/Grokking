import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import json
import argparse

from core.models import GetModel
from core.utils import SaveConfig, SaveModel, LoadModel, PlotModel, DivideDataset
from core.datasets import DatasetCN, DatasetCNFM
from core.engine import TrainEpoch, Validate
from configs import DOT_CNFM


def train():
    N = 4
    config = DOT_CNFM(N)

    device = torch.device("cpu")

    save_path = config["experiment"]["save_path"]
    os.makedirs(save_path, exist_ok=True)
    
    dataset = DatasetCNFM(N)

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

    train_acc_array = []
    val_acc_array = []
    epochs_array = []

    for e in range(config["training"]["epochs"]):
        train_acc = TrainEpoch(model, train_loader, optimizer, device)
        val_acc = Validate(model, val_loader, device)

        train_acc_array.append(train_acc * 100)
        val_acc_array.append(val_acc * 100)
        epochs_array.append(e + 1)

        print(e + 1)

        if (e + 1) % 10 == 0:
            SaveModel(model, optimizer, e + 1, save_path)

        if (e + 1) % 10 == 0:
            PlotModel(epochs_array, train_acc_array, val_acc_array, save_path)


            
if __name__ == "__main__":
    train()