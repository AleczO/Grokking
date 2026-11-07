"""Train a registered grokking experiment (see configs.py)."""
import torch
import torch.optim as optim
import argparse

from core import (GetModel, TrainEpoch, Validate, BuildDatasets, to_tensors,
                  SaveConfig, SaveModel, MetricsLogger, PlotModel,
                  build_config)
from configs import EXPERIMENTS


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--experiment", choices=sorted(EXPERIMENTS))
    p.add_argument("--list", action="store_true", help="List registered experiments and exit")
    p.add_argument("--N", type=int, help="Problem size N (modulus p for DOT_M)")

    p.add_argument("--epochs", type=int, default=20000)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.15, help="AdamW weight decay")
    p.add_argument("--train-frac", type=float, default=0.4,
                   help="Train fraction (random-split datasets only)")
    p.add_argument("--seed", type=int, default=42)

    # Bookkeeping only: not stored in config.json, does not affect results
    p.add_argument("--save-every", type=int, default=500,
                   help="Save a checkpoint and the plot every N epochs")

    return p.parse_args()


def train(config, save_every=500):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    exp = config["experiment"]
    training = config["training"]
    torch.manual_seed(training["seed"])

    save_path = exp["save_path"]
    SaveConfig(config, save_path)

    train_ds, val_ds = BuildDatasets(exp["dataset"], exp["N"], training["train_frac"])
    train_x, train_y = to_tensors(train_ds, device)
    val_x, val_y = to_tensors(val_ds, device)

    model = GetModel(config).to(device)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=training["lr"],
        weight_decay=training["weight_decay"]
    )

    print(f"Starting training: {exp['arch']}_{exp['dataset']} (N={exp['N']}) -> {save_path}")

    logger = MetricsLogger(save_path)

    for epoch in range(1, training["epochs"] + 1):
        train_loss, train_acc = TrainEpoch(model, train_x, train_y, optimizer)
        val_loss, val_acc = Validate(model, val_x, val_y)

        logger.log(epoch=epoch,
                   train_loss=train_loss, train_acc=train_acc * 100,
                   val_loss=val_loss, val_acc=val_acc * 100)

        print(f"epoch {epoch} | train acc {train_acc * 100:.2f}% | val acc {val_acc * 100:.2f}%")

        if epoch % save_every == 0 or epoch == training["epochs"]:
            h = logger.history
            SaveModel(model, optimizer, epoch, save_path)
            PlotModel(h["epoch"], h["train_acc"], h["val_acc"], save_path)

    logger.close()


def main():
    args = parse_args()
    config = build_config(args, EXPERIMENTS)
    if config is not None:
        train(config, save_every=args.save_every)


if __name__ == "__main__":
    main()