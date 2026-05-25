import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime
from torch.utils.data import random_split, DataLoader


from dataset import complexNumbers
from model import GrokkingSetup
from utils import PlotModel, SaveConfig, SaveModel, DivideDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

N = 6

config = {
    "N": N,
    "d_model": 128,
    "voc_size": 4 * N * N + 1,
    "pos_encoding": 4,
    "m_hidden": 128,
    "n_layers": 2,
    "weight_decay": 0.15,
    "epochs": 20000,
    "lr": 1e-3,
    "batch_size": 256,
    "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S")
}
path = SaveConfig(config)


dataset = complexNumbers(N)
train_dataset, val_dataset = DivideDataset(dataset, 0.40)

train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False)


model = GrokkingSetup(
    d_model=config["d_model"], 
    pos_encoding=config["pos_encoding"],
    m_hidden=config["m_hidden"],
    voc_size=config["voc_size"],
    n_layers=config["n_layers"]
)
model.to(device)


lossFN = nn.CrossEntropyLoss()

optimizer = torch.optim.AdamW(
    model.parameters(), 
    lr=config["lr"], 
    weight_decay=config["weight_decay"] 
)

epochs = config["epochs"]

val_acc_array = []
train_acc_array = []
epochs_array = []

print("Starting training...")

for e in range(epochs):

    # Training
    train_acc_re = 0.0
    train_acc_im = 0.0
    train_total_acc = 0.0


    model.train()

    for idx, (inputs, outputs) in enumerate(train_loader):
        inputs, outputs = inputs.to(device), outputs.to(device)
        targets_re = outputs[:, 0]
        targets_im = outputs[:, 1]

        logits_re, logits_im = model(inputs)

        loss_re = lossFN(logits_re, targets_re)
        loss_im = lossFN(logits_im, targets_im)
        loss = loss_re + loss_im

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        preds_re = logits_re.argmax(dim=1) 
        preds_im = logits_im.argmax(dim=1)

        batch_acc_re = (preds_re == targets_re).float().mean().item()
        batch_acc_im = (preds_im == targets_im).float().mean().item()
        batch_total = ((preds_re == targets_re) & (preds_im == targets_im)).float().mean().item()

        train_acc_re += batch_acc_re
        train_acc_im += batch_acc_im
        train_total_acc += batch_total

    train_acc_re = train_acc_re / len(train_loader)
    train_acc_im = train_acc_im / len(train_loader)
    train_total_acc = train_total_acc / len(train_loader)


    # Validation
    val_acc_re = 0.0
    val_acc_im = 0.0
    val_total_acc = 0.0

    model.eval()

    with torch.no_grad():
        for idx, (inputs, outputs) in enumerate(val_loader):
            inputs, outputs = inputs.to(device), outputs.to(device)
            
            targets_re = outputs[:, 0]
            targets_im = outputs[:, 1]

            logits_re, logits_im = model(inputs)

            loss_re = lossFN(logits_re, targets_re)
            loss_im = lossFN(logits_im, targets_im)
            loss = loss_re + loss_im

            preds_re = logits_re.argmax(dim=1) 
            preds_im = logits_im.argmax(dim=1)
                
            batch_acc_re = (preds_re == targets_re).float().mean().item()
            batch_acc_im = (preds_im == targets_im).float().mean().item()
            batch_total = ((preds_re == targets_re) & (preds_im == targets_im)).float().mean().item()

            val_acc_re += batch_acc_re
            val_acc_im += batch_acc_im
            val_total_acc += batch_total

    val_acc_re = val_acc_re / len(val_loader)
    val_acc_im = val_acc_im / len(val_loader)
    val_total_acc = val_total_acc / len(val_loader)



    train_acc_array.append(train_total_acc * 100)
    val_acc_array.append(val_total_acc * 100)
    epochs_array.append(e)


    if (e + 1) % 100 == 0:
        print(f"Epoch {e + 1}/{epochs} | Train Acc: {train_total_acc*100:.2f}% - Re Acc: {train_acc_re*100:.2f}%, Im Acc: {train_acc_im*100:.2f}%  | Val Acc: {val_total_acc*100:.2f}  -  % Re Acc: {val_acc_re*100:.2f}, Im Acc: {val_acc_im*100:.2f}")

    if (e + 1) % 1000 == 0:
        SaveModel(model, e + 1, path)

    if (e + 1) % 500 == 0:
        PlotModel(epochs_array, train_acc_array, val_acc_array, path)
        

print(f"Training finished. All results in {path}")