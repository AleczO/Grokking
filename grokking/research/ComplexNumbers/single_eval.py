import torch
import argparse
import json
import os
from model import GrokkingSetup
from dataset import complexNumbers
from torch.utils.data import DataLoader, random_split

def run_single(ckpt_path):
    run_dir = os.path.dirname(os.path.dirname(ckpt_path))
    config_path = os.path.join(run_dir, "config.json")
    
    if not os.path.exists(config_path):
        print(f"Błąd: Nie znaleziono pliku config w {config_path}")
        return

    with open(config_path, "r") as f:
        cfg = json.load(f)


    device = torch.device("cpu")
    

    model = GrokkingSetup(
        d_model=cfg["d_model"], 
        voc_size=cfg["voc_size"],
        pos_encoding=cfg.get("pos_encoding", 4), 
        d_k=cfg["d_k"], 
        d_v=cfg["d_v"], 
        n_layers=cfg["n_layers"]
    )
    
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    model.to(device).eval()

    N = 5 
    dataset = complexNumbers(N)
    

    dataset_size = len(dataset)
    train_size = int(0.4 * dataset_size) 
    val_size = dataset_size - train_size
    
    generator = torch.Generator().manual_seed(42)
    _, val_dataset = random_split(dataset, [train_size, val_size], generator=generator)
    val_loader = DataLoader(val_dataset, batch_size=cfg.get("batch_size", 512), shuffle=False)


    correct_re, correct_im, correct_total = 0, 0, 0
    total = 0

    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            logits_re, logits_im = model(inputs)
            
            preds_re = logits_re.argmax(dim=1)
            preds_im = logits_im.argmax(dim=1)
            
            targets_re = targets[:, 0]
            targets_im = targets[:, 1]

            total += targets.size(0)
            
            correct_re += (preds_re == targets_re).sum().item()
            correct_im += (preds_im == targets_im).sum().item()
            
            correct_total += ((preds_re == targets_re) & (preds_im == targets_im)).sum().item()

    acc_re = 100. * correct_re / total
    acc_im = 100. * correct_im / total
    acc_total = 100. * correct_total / total
    
    print("-" * 30)
    print(f"Checkpoint: {os.path.basename(ckpt_path)}")
    print(f"Total Accuracy: {acc_total:.2f}%")
    print(f"Real Part Acc: {acc_re:.2f}%")
    print(f"Imag Part Acc: {acc_im:.2f}%")
    print("-" * 30)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=True, help="Sciezka do pliku .pt")
    args = parser.parse_args()
    run_single(args.path)