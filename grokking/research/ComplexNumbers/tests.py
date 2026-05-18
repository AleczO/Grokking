import os
import json
import torch
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from model import GrokkingSetup


def GetModelFromConfig(path):
    config_path = os.path.join(path, "config.json")
    with open(config_path, "r") as f:
        config = json.load(f)
        
    model = GrokkingSetup(
        d_model=config["d_model"], 
        voc_size=config["voc_size"],
        pos_encoding=config.get("pos_encoding", 4), 
        d_k=config["d_k"], 
        d_v=config["d_v"], 
        n_layers=config["n_layers"]
    )
    
    return model


def embedding_analysis_tsne(model, N=5, offset=50):
    weights = model.tokens_embed.weight.detach().cpu().numpy()
    num_tokens = weights.shape[0]

    tsne = TSNE(
        n_components=2, 
        learning_rate='auto', 
        init='pca', 
        perplexity=15, 
    )
    embeds_2d = tsne.fit_transform(weights)

    df = pd.DataFrame({
        'x': embeds_2d[:, 0],
        'y': embeds_2d[:, 1],
        'val': [i - offset for i in range(num_tokens)]
    })

    plt.figure(figsize=(12, 9))

    sc = plt.scatter(
        df.x, df.y, 
        c=df.val, 
        cmap="magma", 
        s=100, 
        linewidth=0.5,
        alpha=0.8
    )

    cbar = plt.colorbar(sc)
    cbar.set_label('Wartość (token - offset)', fontsize=12)

    for i in range(0, num_tokens, 5):
        plt.text(
            df.x[i], df.y[i], 
            str(int(df.val[i])), 
            fontsize=9, weight='bold', alpha=0.7
        )

    plt.title(f"t-SNE For Tokens Embeddings (N={N})", fontsize=15)
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.show()



path = r"/home/aleczo/Documents/Temp/Research/grokking/research/ComplexNumbers/experiments/run_L2_D64_20260510_201635"

model = GetModelFromConfig(path)

model_path = r"/home/aleczo/Documents/Temp/Research/grokking/research/ComplexNumbers/experiments/run_L2_D64_20260510_201635/checkpoints/model_epoch_18999.pt"

model.load_state_dict(torch.load(model_path))

embedding_analysis_tsne(model, N=5)