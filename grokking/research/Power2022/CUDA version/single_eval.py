import torch
import argparse
import json
import os
from model import GrokkingSetup
from dataset import ModuloDataset
from torch.utils.data import DataLoader, random_split

# Tu wstaw swoją logikę GetModelFromConfig lub wczytaj config ręcznie
def run_single(ckpt_path):
    # 1. Wyciągnij parametry z folderu nadrzędnego (zakładając strukturę experiments/run/checkpoints)
    run_dir = os.path.dirname(os.path.dirname(ckpt_path))
    with open(os.path.join(run_dir, "config.json"), "r") as f:
        cfg = json.load(f)

    # 2. Setup (FORCE CPU, żeby wykluczyć błędy sterownika AMD)
    device = torch.device("cpu")
    model = GrokkingSetup(p=97, d_model=cfg["d_model"], d_k=cfg["d_k"], d_v=cfg["d_v"], n_layers=cfg["n_layers"])
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    model.to(device).eval()

    # 3. Data (zawsze ten sam seed!)
    dataset = ModuloDataset(97)
    generator = torch.Generator().manual_seed(42)
    _, val_dataset = random_split(dataset, [int(0.2*97*97), 97*97 - int(0.2*97*97)], generator=generator)
    val_loader = DataLoader(val_dataset, batch_size=512)

    # 4. Eval
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, outputs in val_loader:
            logits = model(inputs)
            _, predicted = logits.max(1)
            total += outputs.size(0)
            correct += (predicted == outputs).sum().item()

    acc = 100. * correct / total
    print(f"{os.path.basename(ckpt_path)}: {acc:.2f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str)
    args = parser.parse_args()
    run_single(args.path)