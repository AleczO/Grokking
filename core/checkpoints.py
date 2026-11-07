"""Run directories, config files and model checkpoints on disk."""
import json
import os
import re

import torch


def NextRunDir(base_dir):
    """Return the next auto-numbered run directory: <base_dir>/exp_XXX."""
    os.makedirs(base_dir, exist_ok=True)
    existing = [int(m.group(1)) for d in os.listdir(base_dir)
                if (m := re.fullmatch(r"exp_(\d+)", d))]
    next_id = max(existing, default=0) + 1
    return os.path.join(base_dir, f"exp_{next_id:03d}")


def SaveConfig(config, path):
    """Write the run config to <path>/config.json."""
    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, "config.json"), "w") as f:
        json.dump(config, f, indent=4)
    return path


def SaveModel(model, optimizer, epoch, path):
    """Save model and optimizer state to <path>/checkpoint_epoch_<epoch>.pt."""
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    torch.save(checkpoint, os.path.join(path, f"checkpoint_epoch_{epoch}.pt"))


def LoadModel(model, optimizer, path, device):
    """Load a checkpoint into model and optimizer; returns its epoch."""
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint["epoch"]