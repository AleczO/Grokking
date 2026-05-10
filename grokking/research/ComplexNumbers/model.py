import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np

class MLP(nn.Module):
    def __init__(self, D, scale_factor=4) -> None:
        super().__init__()
        Hidden_D = scale_factor * D 

        self.FFN = nn.Sequential(
            nn.Linear(D, Hidden_D),
            nn.ReLU(),
            nn.Linear(Hidden_D, D)
        )

    def forward(self, X):
        return self.FFN(X)  

class DecoderLayer(nn.Module):
    def __init__(self, D, D_k, D_v) -> None:
        super().__init__()
        self.W_q = nn.Parameter(torch.normal(0, 0.02, size=(D, D_k)))
        self.W_k = nn.Parameter(torch.normal(0, 0.02, size=(D, D_k)))
        self.W_v = nn.Parameter(torch.normal(0, 0.02, size=(D, D_v)))
        self.D_k = float(D_k)

        self.MLP = MLP(D)
        self.LN1 = nn.LayerNorm(D)
        self.LN2 = nn.LayerNorm(D)

        self.W_o = nn.Linear(D_v, D)

    def Attention(self, X):
        Q = X @ self.W_q 
        K = X @ self.W_k 
        V = X @ self.W_v 

        scores = Q @ K.transpose( -2, -1) / np.sqrt(self.D_k)
        weights = torch.softmax(scores, dim=-1) 
        attn = weights @ V  

        return attn

    def forward(self, X):
        Attn = self.Attention(self.LN1(X))

        Z = self.W_o(Attn) + X

        X_n = self.MLP(self.LN2(Z)) + Z

        return X_n


class GrokkingSetup(nn.Module):
    def __init__(self, d_model, pos_encoding, voc_size, d_k, d_v, n_layers):
        super().__init__()
        self.vocab = voc_size
        self.d_model = d_model
        self.pos_encoding = pos_encoding


        self.tokens_embed = nn.Embedding(self.vocab, d_model)
        self.pos_embed = nn.Embedding(self.pos_encoding, d_model)

        self.layers = nn.ModuleList([
            DecoderLayer(d_model, d_k, d_v) for _ in range(n_layers)
        ])

        
        self.unembed_re = nn.Linear(self.d_model, self.vocab)
        self.unembed_im = nn.Linear(self.d_model, self.vocab)

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

    def forward(self, x):
        batch_size, seq_len = x.shape

        pos = torch.arange(seq_len, device=x.device).unsqueeze(0)

        X = self.tokens_embed(x) + self.pos_embed(pos)

        for layer in self.layers:
            X = layer(X)

            
        out = X[:, -1, :]

        logits_re = self.unembed_re(out)
        logits_im = self.unembed_im(out)

        return logits_re, logits_im