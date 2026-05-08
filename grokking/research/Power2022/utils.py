import seaborn as sns
import matplotlib.pyplot as plt

sns.set_theme(style="whitegrid", context="talk")

def PlotModel(epochs, train_acc_arr, val_acc_arr):
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
    plt.savefig("plots/grokking_pro_look.png", dpi=300)
    plt.close()