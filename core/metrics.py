"""Per-epoch metrics: CSV logging and accuracy plot."""
import csv
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

METRICS_FIELDS = ["epoch", "train_loss", "train_acc", "val_loss", "val_acc"]


class MetricsLogger:
    """Appends one row of metrics per epoch to <path>/metrics.csv.

    Rows are flushed immediately, so the file is complete up to the last
    finished epoch even if training is interrupted.
    """

    def __init__(self, path):
        self.file = open(os.path.join(path, "metrics.csv"), "w", newline="")
        self.writer = csv.DictWriter(self.file, fieldnames=METRICS_FIELDS)
        self.writer.writeheader()
        self.history = {k: [] for k in METRICS_FIELDS}

    def log(self, **row):
        self.writer.writerow(row)
        self.file.flush()
        for k in METRICS_FIELDS:
            self.history[k].append(row[k])

    def close(self):
        self.file.close()


def PlotModel(epochs, train_acc, val_acc, path):
    """Save the train/validation accuracy curve to <path>/acc_plot.png."""
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

    plt.savefig(os.path.join(path, "acc_plot.png"), dpi=300, bbox_inches="tight")
    plt.close("all")