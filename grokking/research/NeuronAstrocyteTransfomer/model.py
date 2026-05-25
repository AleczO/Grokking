import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionNA(nn.Module):
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

        H = torch.zeros(batch_size, self.d, self.m, device=X.device)
        g = torch.zeros(batch_size, self.m, device=X.device)

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
            H = H * self.gamma + outer
            g = g + h_k
            
        return torch.cat(outputs, dim=1)
    
class LayerAttentionNA(nn.Module):
    def __init__(self, d_model, m_hidden) -> None:
        super().__init__()
        self.Attention = AttentionNA(d_model, m_hidden)
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
    
class GrokkingSetup(nn.Module):
    def __init__(self, d_model, pos_encoding, voc_size, m_hidden, n_layers):
        super().__init__()
        self.vocab = voc_size
        self.d_model = d_model
        self.pos_encoding = pos_encoding

        self.tokens_embed = nn.Embedding(self.vocab, d_model)
        self.pos_embed = nn.Embedding(self.pos_encoding, d_model)

        self.layers = nn.ModuleList([
            LayerAttentionNA(d_model, m_hidden) for _ in range(n_layers)
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
    





class AttenOnlyLayer(nn.Module):
    def __init__(self, d_model, m_hidden) -> None:
        super().__init__()
        self.Attention = AttentionNA(d_model, m_hidden)
        self.LN1 = nn.LayerNorm(d_model)
        self.LN2 = nn.LayerNorm(d_model)


    def forward(self, X):
        Z = self.Attention(self.LN1(X)) + X
        return Z

class AttnOnlySetup(nn.Module):
    def __init__(self, d_model, pos_encoding, voc_size, m_hidden, n_layers):
        super().__init__()
        self.vocab = voc_size
        self.d_model = d_model
        self.pos_encoding = pos_encoding

        self.tokens_embed = nn.Embedding(self.vocab, d_model)
        self.pos_embed = nn.Embedding(self.pos_encoding, d_model)

        self.layers = nn.ModuleList([
            LayerAttentionNA(d_model, m_hidden) for _ in range(n_layers)
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