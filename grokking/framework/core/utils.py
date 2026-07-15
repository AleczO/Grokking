import os
import re
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import json

def SaveConfig(config, path): # Save model configuration
    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, 'config.json'), 'w') as f:
        json.dump(config, f, indent=4)
    return path

def SaveModel(model, optimizer, epoch, path): # Save model parameters
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    torch.save(checkpoint, os.path.join(path, f"checkpoint_epoch_{epoch}.pt"))

def LoadModel(model, optimizer, path, device): # Load model parameters
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['epoch']

def PlotModel(epochs, train_acc, val_acc, path): # Plot epochs
    sns.set_theme(style="whitegrid")
    
    df = pd.DataFrame({
        "Epoch": epochs,
        "Train": train_acc,
        "Validation": val_acc
    })
    
    df_melted = df.melt("Epoch", var_name="Dataset", value_name="Accuracy")
    
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df_melted, x="Epoch", y="Accuracy", hue="Dataset", linewidth=2.5)
    
    plt.title("Training Process", fontsize=16)
    plt.ylabel("Accuracy (%)", fontsize=12)
    plt.xlabel("Epochs", fontsize=12)
    plt.ylim(0, 105)
    
    plt.savefig(os.path.join(path, "acc_plot.png"), dpi=300, bbox_inches='tight')
    plt.close('all')

def DivideDataset(dataset, train_frac): # Divide dataset for training
    train_size = int(train_frac * len(dataset))
    val_size = len(dataset) - train_size
    return torch.utils.data.random_split(dataset, [train_size, val_size])

def NextRunDir(base_dir): # Auto-increment exp_XXX run directory
    os.makedirs(base_dir, exist_ok=True)
    existing = [int(m.group(1)) for d in os.listdir(base_dir)
                if (m := re.fullmatch(r"exp_(\d+)", d))]
    next_id = max(existing, default=0) + 1
    return os.path.join(base_dir, f"exp_{next_id:03d}")