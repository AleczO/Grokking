import torch
import argparse
import json
import os
from model import GrokkingSetup
from dataset import complexNumbers # Upewnij się, że ten plik jest identyczny jak ten w treningu
from torch.utils.data import DataLoader

def run_single(ckpt_path):
    run_dir = os.path.dirname(os.path.dirname(ckpt_path))
    config_path = os.path.join(run_dir, "config.json")
    
    if not os.path.exists(config_path):
        print(f"Błąd: Nie znaleziono configu w {config_path}")
        return

    with open(config_path, "r") as f:
        config = json.load(f)

    device = torch.device("cpu")
    
    model = GrokkingSetup(
        d_model=config["d_model"], 
        voc_size=config["voc_size"],
        pos_encoding=config.get("pos_encoding", 4), 
        m_hidden=config["m_hidden"], 
        n_layers=config["n_layers"]
    )
    
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    model.to(device).eval()

    N = config["N"]
    dataset = complexNumbers(N)

    loader = DataLoader(dataset, batch_size=config.get("batch_size", 512), shuffle=False)

    correct_total = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            logits_re, logits_im = model(inputs)
            
            preds_re = logits_re.argmax(dim=1)
            preds_im = logits_im.argmax(dim=1)
            
            targets_re = targets[:, 0]
            targets_im = targets[:, 1]

            total += targets.size(0)
            correct_total += ((preds_re == targets_re) & (preds_im == targets_im)).sum().item()

    acc_total = 100. * correct_total / total
    
    print("-" * 30)
    print(f"Checkpoint: {os.path.basename(ckpt_path)}")
    print(f"Total Accuracy: {acc_total:.2f}%")
    print("-" * 30)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=True)
    args = parser.parse_args()
    run_single(args.path)