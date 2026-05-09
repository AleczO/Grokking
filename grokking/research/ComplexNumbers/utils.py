import os
import json
import torch
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_theme(style="whitegrid", context="talk")

def SaveConfig(config):
    run_name = f"run_L{config['n_layers']}_D{config['d_model']}_{config['timestamp']}"
    run_path = os.path.join("experiments", run_name)
    
    os.makedirs(os.path.join(run_path, "plots"), exist_ok=True)
    os.makedirs(os.path.join(run_path, "checkpoints"), exist_ok=True)

    with open(os.path.join(run_path, "config.json"), "w") as f:
        json.dump(config, f, indent=4)

    print(f"Eksperyment zainicjalizowany w: {run_path}")
    return run_path


def PlotModel(epochs, train_acc_arr, val_acc_arr, run_path):
    plt.figure(figsize=(12, 7))
    palette = sns.color_palette("viridis", 2)
    
    sns.lineplot(x=epochs, y=train_acc_arr, label="Train Accuracy", color=palette[0], linewidth=2)
    sns.lineplot(x=epochs, y=val_acc_arr, label="Validation Accuracy", color=palette[1], linewidth=2.5)
    
    plt.title("Grokking Progress: Modular Addition", pad=20)
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy (%)")
    plt.ylim(-5, 105)
    
    plt.fill_between(epochs, val_acc_arr, alpha=0.1, color=palette[1])
    
    plt.tight_layout()

    save_path = os.path.join(run_path, "plots", "epochs.png")
    plt.savefig(save_path, dpi=300)
    plt.close()


def SaveModel(model, epoch, run_path):
    checkpoint_dir = os.path.join(run_path, "checkpoints")
    path_to_checkpoint = os.path.join(checkpoint_dir, f"model_epoch_{epoch}.pt")
    
    torch.save(model.state_dict(), path_to_checkpoint)