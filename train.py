import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse

from core.models import GetModel
from core.utils import SaveConfig, SaveModel, PlotModel, NextRunDir
from core.datasets import BuildDatasets
from core.engine import TrainEpoch, Validate
from configs import EXPERIMENTS


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", choices=sorted(EXPERIMENTS))
    parser.add_argument("--list", action="store_true", help="List registered experiments and exit")
    parser.add_argument("--N", type=int, help="Problem size N")
    parser.add_argument("--epochs", type=int, help="Override epoch count from the experiment config")
    return parser.parse_args()


def train():
    args = parse_args()

    if args.list or not args.experiment:
        print("Available experiments:")
        for name in sorted(EXPERIMENTS):
            print(f"  {name}")
        return

    if args.N is None:
        raise SystemExit("--N is required to run an experiment")

    config = EXPERIMENTS[args.experiment](args.N)
    if args.epochs:
        config["training"]["epochs"] = args.epochs
    config["experiment"]["save_path"] = NextRunDir(config["experiment"]["save_path"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    save_path = config["experiment"]["save_path"]
    os.makedirs(save_path, exist_ok=True)

    SaveConfig(config, save_path)

    train_ds, val_ds = BuildDatasets(config["experiment"]["dataset"], args.N, config["training"]["train_frac"])
    train_loader = DataLoader(train_ds, batch_size=config["training"]["batch_size"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config["training"]["batch_size"], shuffle=False)

    model = GetModel(config)
    model.to(device)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=config["training"]["lr"],
        weight_decay=config["training"]["weight_decay"]
    )

    print(f"Starting training: {args.experiment} (N={args.N}) -> {save_path}")

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
