import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np




# ---------------------------------------------------------------------- Neuron-Astrocyte Transformer Implementation --------------------------------------------------

class AttentionNAT(nn.Module): # Attention Neuron-Astrocyte Transformer 
    def __init__(self, d_model, m_hidden, gamma=0.95):
        super().__init__()
        self.d = d_model
        self.m = m_hidden
        self.gamma = gamma

        self.W_q = nn.Parameter(torch.randn(self.m, self.d) * 0.02)
        self.W_k = nn.Parameter(torch.randn(self.m, self.d) * 0.02)
        
        self.register_buffer('b', torch.empty(self.m).uniform_(0, 2 * torch.pi))

    def _phi_q(self, x):
        x = torch.nn.functional.normalize(x, p=2, dim=-1)
        return torch.cos(x @ self.W_q.t() + self.b) 

    def _phi_k(self, x):
        x = torch.nn.functional.normalize(x, p=2, dim=-1)
        return torch.cos(x @ self.W_k.t() + self.b)

    def forward(self, X):
        if X.dim() == 2:
            X = X.unsqueeze(1)

        batch_size, seq_len, _ = X.shape

        H = torch.zeros(batch_size, self.d, self.m, dtype=X.dtype, device=X.device)
        g = torch.zeros(batch_size, self.m, dtype=X.dtype, device=X.device)

        outputs = []

        for t in range(seq_len):
            x_t = X[:, t, :]
            
            h_q = self._phi_q(x_t)
            h_k = self._phi_k(x_t)
            
            if t > 0:
                p = (g * h_q).sum(dim=-1) / self.m
                p = torch.clamp(p, min=1e-4).unsqueeze(-1)
                
                H_h = torch.bmm(H, h_q.unsqueeze(-1)).squeeze(2)
                out_t = (1 / p) * H_h + x_t
            else:
                out_t = x_t
                
            outputs.append(out_t.unsqueeze(1))
            
            outer = torch.bmm(x_t.unsqueeze(-1), h_k.unsqueeze(1)) / self.m
            H = H.clone() * self.gamma + outer
            g = g.clone() + h_k
            
        return torch.cat(outputs, dim=1)
    
class LayerNAT(nn.Module): # Layer of Neuron-Astrocyte Transformer
    def __init__(self, d_model, m_hidden) -> None:
        super().__init__()
        self.Attention = AttentionNAT(d_model, m_hidden)
        self.LN1 = nn.LayerNorm(d_model)
        self.LN2 = nn.LayerNorm(d_model)

        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Linear(d_model * 4, d_model)
        )

    def forward(self, X):
        Z = self.Attention(self.LN1(X)) + X

        Out = self.mlp(self.LN2(Z)) + Z
        return Out
    
class BaseNAT(nn.Module): # Neuron-Astrocyte Transformer
    def __init__(self, d_model, pos_encoding, voc_size, m_hidden, n_layers):
        super().__init__()
        self.vocab = voc_size
        self.d_model = d_model
        self.pos_encoding = pos_encoding

        self.tokens_embed = nn.Embedding(self.vocab, d_model)
        self.pos_embed = nn.Embedding(self.pos_encoding, d_model)

        self.layers = nn.ModuleList([
            LayerNAT(d_model, m_hidden) for _ in range(n_layers)
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
    
    
    def get_hidden_states(self, x):
        x = self.tokens_embed(x) 

        for layer in self.layers:
            x = layer(x)
            
        return x 


# ---------------------------------------------------------------------- Decoder-Only Single-Head Transformer Implementation ------------------------------------------------------

class LayerDOT(nn.Module): # Layer of Decoder-Only Single-Head Transformer
    def __init__(self, D, D_k, D_v, scale_factor=4) -> None:
        super().__init__()
        self.W_q = nn.Parameter(torch.normal(0, 0.02, size=(D, D_k)))
        self.W_k = nn.Parameter(torch.normal(0, 0.02, size=(D, D_k)))
        self.W_v = nn.Parameter(torch.normal(0, 0.02, size=(D, D_v)))
        self.D_k = float(D_k)

        self.LN1 = nn.LayerNorm(D)
        self.LN2 = nn.LayerNorm(D)

        self.W_o = nn.Linear(D_v, D)

        self.mlp = nn.Sequential(
            nn.Linear(D, D * scale_factor),
            nn.ReLU(),
            nn.Linear(D * scale_factor, D)
        )

    def Attention(self, X):
        Q = X @ self.W_q 
        K = X @ self.W_k 
        V = X @ self.W_v 

        scores = Q @ K.transpose(-2, -1) / np.sqrt(self.D_k)
        weights = torch.softmax(scores, dim=-1) 
        attn = weights @ V  

        return attn

    def forward(self, X):
        Attn = self.Attention(self.LN1(X))
        Z = self.W_o(Attn) + X

        X_n = self.mlp(self.LN2(Z)) + Z

        return X_n

class BaseDOT(nn.Module): # Decoder-Only Single-Head Transformer
    def __init__(self, voc_size, pos_encoding, d_model, d_k, d_v, n_layers, num_outputs):
        super().__init__()
        self.p = voc_size
        self.d_model = d_model

        self.tokens_embed = nn.Embedding(voc_size, d_model)
        self.pos_embed = nn.Embedding(pos_encoding, d_model)

        self.layers = nn.ModuleList([
            LayerDOT(d_model, d_k, d_v) for _ in range(n_layers)
        ])

        self.output_heads = nn.ModuleList([
            nn.Linear(d_model, voc_size) for _ in range(num_outputs)
        ])

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Embedding, nn.Linear)):
                nn.init.normal_(m.weight, std=0.02)

    def forward(self, x):
        batch_size, seq_len = x.shape

        pos = torch.arange(seq_len, device=x.device).unsqueeze(0)

        X = self.tokens_embed(x) + self.pos_embed(pos)

        for layer in self.layers:
            X = layer(X)

        out = X[:, -1, :]

        logits = self.unembed(out)
        return logits

# ---------------------------------------------------------------------- Decoder-Only Multi-Head Transformer Implementation ------------------------------------------------------

class AttentionMHDOT(nn.Module): # Attention of Decoder-Only Multi-Head Transformer
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, X):
        batch_size, seq_len, _ = X.shape

        Q = self.W_q(X).view(batch_size, seq_len, self.n_heads, self.d_head).transpose(-2, -1)
        K = self.W_k(X).view(batch_size, seq_len, self.n_heads, self.d_head).transpose(-2, -1)
        V = self.W_v(X).view(batch_size, seq_len, self.n_heads, self.d_head).transpose(-2, -1)

        scores = Q @ K.transpose(-2, -1) / np.sqrt(self.d_head)
        weights = torch.softmax(scores, dim=-1)
        
        attn = (weights @ V).transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)

        return self.W_o(attn)

class LayerMHDOT(nn.Module): # Layer of Decoder-Only Multi-Head Transformer
    def __init__(self, D, n_heads, scale_factor=4):
        super().__init__()

        self.MHA = AttentionMHDOT(D, n_heads)

        self.LN1 = nn.LayerNorm(D)
        self.LN2 = nn.LayerNorm(D)

        self.mlp = nn.Sequential(
            nn.Linear(D, D * scale_factor),
            nn.ReLU(),
            nn.Linear(D * scale_factor, D)
        )

    def forward(self, X):
        Attn = self.MHA(self.LN1(X))
        Z = Attn + X

        Out = self.mlp(self.LN2(Z)) + Z

        return Out

class BaseMHDOT(nn.Module): # Decoder-Only Multi-Head Transformer 
    def __init__(self, voc_size, pos_encoding, d_model, n_heads, n_layers, num_outputs=2):
        super().__init__()
        self.d_model = d_model
        self.voc_size = voc_size
        self.pos_encoding = pos_encoding

        self.tokens_embed = nn.Embedding(voc_size, d_model)
        self.pos_embed = nn.Embedding(pos_encoding, d_model)

        self.layers = nn.ModuleList([
            LayerMHDOT(d_model, n_heads) for _ in range(n_layers)
        ])

        self.output_heads = nn.ModuleList([
            nn.Linear(d_model, voc_size) for _ in range(num_outputs)
        ])

        self.init_weights()

    def forward(self, x):
        pos = torch.arange(x.shape[1], device=x.device).unsqueeze(0)
        X = self.tokens_embed(x) + self.pos_embed(pos)
        
        for layer in self.layers:
            X = layer(X)
            
        out = X[:, -1, :]
        return [head(out) for head in self.output_heads]



# --------------------------------------------------------------- Loss Functions --------------------------------------------------------------


class DOT(BaseDOT):
    def computeLoss(self, logits_list, targets):
        total_loss = 0.0
        total_acc = 0.0
        
        if targets.dim() == 1:
            targets = targets.unsqueeze(1)
            
        for i, logits in enumerate(logits_list):
            loss = F.cross_entropy(logits, targets[:, i])
            acc = (logits.argmax(dim=1) == targets[:, i]).float().mean().item()
            
            total_loss += loss
            total_acc += acc

        return total_loss, total_acc / len(logits_list)
    
    def computeLoss(self, logits_list, targets):
        total_loss = 0.0
        
        if targets.dim() == 1:
            targets = targets.unsqueeze(1)
            
        num_outputs = targets.size(1)
        
        masks = []
        
        for i in range(num_outputs):
            logits = logits_list[i]
            
            loss = F.cross_entropy(logits, targets[:, i])
            total_loss += loss
            
            is_head_correct = logits.argmax(dim=1) == targets[:, i]
            masks.append(is_head_correct)
            
        scaled_loss = total_loss / num_outputs

        correct = masks[0]
        for mask in masks[1:]:
            correct = correct & mask
            
        sequence_acc = correct.float().mean().item()

        return scaled_loss, sequence_acc

class NAT(BaseNAT):
    def computeLoss(self, logits, targets):
        logits_re, logits_im = logits
        
        loss_re = F.cross_entropy(logits_re, targets[:, 0])
        loss_im = F.cross_entropy(logits_im, targets[:, 1])
        loss = loss_re + loss_im
        
        preds_re = logits_re.argmax(dim=1)
        preds_im = logits_im.argmax(dim=1)
        
        acc = ((preds_re == targets[:, 0]) & (preds_im == targets[:, 1])).float().mean().item()
        
        return loss, acc

class MHDOT(BaseMHDOT):
    def computeLoss(self, logits_list, targets):
        total_loss = 0.0
        total_acc = 0.0
        
        for i, logits in enumerate(logits_list):
            loss = F.cross_entropy(logits, targets[:, i])
            
            acc = (logits.argmax(dim=1) == targets[:, i]).float().mean().item()
            
            total_loss += loss
            total_acc += acc

        return total_loss, total_acc / len(logits_list)



# ---------------------------------------------------------------------- Model Selection ------------------------------------------------------


def GetModel(config):
    setup = config["setup"]
    arch = config["experiment"]["arch"]

    if arch == "NAT": # d_model, pos_encoding, voc_size, m_hidden, n_layers
        return NAT(
            d_model=setup["d_model"],
            pos_encoding=setup["pos_encoding"],
            voc_size=setup["voc_size"],
            m_hidden=setup["m_hidden"],
            n_layers=setup["n_layers"]
        )
    if arch == "DOT":
        return DOT(
            voc_size=setup["voc_size"],
            d_model=setup["d_model"],
            pos_encoding=setup["pos_encoding"],
            d_k=setup["d_k"], 
            d_v=setup["d_v"],
            n_layers=setup["n_layers"],
            num_outputs=setup["num_outputs"]
        )
    if arch == "MHDOT":
        return MHDOT(
            voc_size=setup["voc_size"],
            pos_encoding=setup["pos_encoding"],
            d_model=setup["d_model"],
            n_heads=setup["n_heads"],
            n_layers=setup["n_layers"],
            num_outputs=setup["num_outputs"],
        )
        
    return -1

