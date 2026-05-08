import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime
from torch.utils.data import random_split, DataLoader

from dataset import ModuloDataset
from model import GrokkingSetup
from utils import PlotModel, SaveConfig, SaveModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

config = {
    "d_model": 16,
    "d_k": 128,
    "d_v": 128,
    "n_layers": 3,
    "weight_decay": 1.0,
    "epochs": 20000,
    "lr": 1e-3,
    "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S")
}
path = SaveConfig(config)



P = 97
N = P * P
train_N = int(0.20 * N)
val_N = N - train_N


dataset = ModuloDataset(P)
train_dataset, val_dataset = random_split(dataset, [train_N, val_N])

train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)

model = GrokkingSetup(
    p=P, 
    d_model=config["d_model"], 
    d_k=config["d_k"], 
    d_v=config["d_v"], 
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
    train_total = 0
    train_correct = 0

    model.train()

    for idx, (inputs, outputs) in enumerate(train_loader):
        inputs, outputs = inputs.to(device), outputs.to(device)
        optimizer.zero_grad()

        logits = model(inputs)
        loss = lossFN(logits, outputs)

        loss.backward()
        optimizer.step()

        _, predicted = logits.max(1)
        train_correct += (predicted == outputs).sum().item()
        train_total += outputs.size(0)

    train_acc = 100. * train_correct / train_total


    # Validation
    total = 0
    correct = 0

    model.eval()

    with torch.no_grad():
        for idx, (inputs, outputs) in enumerate(val_loader):
            inputs, outputs = inputs.to(device), outputs.to(device)

            logits = model(inputs)
            loss = lossFN(logits, outputs)

            _, predicted = logits.max(1)
                
            total += outputs.size(0)
            correct += (predicted == outputs).sum().item()

    val_acc = 100. * correct / total


    train_acc_array.append(train_acc)
    val_acc_array.append(val_acc)
    epochs_array.append(e)


    if (e + 1) % 100 == 0:
        print(f"Epoch {e + 1}/{epochs} | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")

    if e % 1000 == 0:
        SaveModel(model, e, path)

    if e % 500 == 0:
        PlotModel(epochs_array, train_acc_array, val_acc_array, path)
        

SaveModel(model, e, path)
PlotModel(epochs_array, train_acc_array, val_acc_array, path)
print(f"Training finished. All results in {path}")