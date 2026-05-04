import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import random_split, DataLoader
from dataset import ModuloDataset
from model import GrokkingSetup
from utils import PlotModel

folder_plots = "plots"
folder_checkpoints = "checkpoints"


os.environ['HSA_OVERRIDE_GFX_VERSION'] = '10.3.0'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

P = 97

dataset = ModuloDataset(P)

N = P * P
train_N = int(0.20 * N)
val_N = N - train_N


train_dataset, val_dataset = random_split(dataset, [train_N, val_N])

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)

model = GrokkingSetup(p=P, d_model=128, d_k=128, d_v=128, n_layers=2)
model.to(device)

lossFN = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.8)


epochs = 20000

val_acc = 0
train_acc = 0 

val_acc_array = []
train_acc_array = []
epochs_array = []

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
        print(f"Epochs: {e + 1}")

    if e % 1000 == 0:
        os.makedirs(folder_plots, exist_ok=True)
        os.makedirs(folder_checkpoints, exist_ok=True)

        path_to_checkpoint = os.path.join(folder_checkpoints, f"model_epoch_{e}.pt")
        torch.save(model.state_dict(), path_to_checkpoint)


    if e % 500 == 0:
        PlotModel(epochs_array, train_acc_array, val_acc_array)
        

PlotModel(epochs_array, train_acc_array, val_acc_array)