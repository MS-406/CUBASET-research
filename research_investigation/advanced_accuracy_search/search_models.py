"""
research_investigation/advanced_accuracy_search/search_models.py
---------------------------------------------------------------
Defines the modular temporal architectures for the Advanced Accuracy & Pareto Search:
  1. MultiScaleConvAE (Parallel Multi-Kernel 1D Convolutions: k=3, 7, 11, 15)
  2. DilatedTCN_AE (Dilated Temporal Convolutional Network with residual connections)
  3. TinyGRU_AE (Gated Recurrent Autoencoder with bidirectional encoder)
  4. LightTransformerAE (Compact 1-Layer, 2-Head Self-Attention Autoencoder)
  5. HybridPredictiveAE (Dual-Head Architecture: Reconstruction + Next-Step Forecasting)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# -------------------------------------------------------------------------
# 1. MultiScaleConvAE
# -------------------------------------------------------------------------
class MultiScaleConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernels=[3, 7, 11, 15]):
        super(MultiScaleConvBlock, self).__init__()
        num_k = len(kernels)
        ch_div = max(1, out_channels // num_k)
        rem = out_channels - (num_k - 1) * ch_div
        
        self.convs = nn.ModuleList()
        for idx, k in enumerate(kernels):
            c_out = rem if idx == num_k - 1 else ch_div
            self.convs.append(nn.Conv1d(in_channels, c_out, kernel_size=k, padding=k // 2))
        self.act = nn.ReLU()

    def forward(self, x):
        outputs = [c(x) for c in self.convs]
        return self.act(torch.cat(outputs, dim=1))

class MultiScaleConvAE(nn.Module):
    def __init__(self, n_features=1, hidden_dim=16, latent_dim=8, kernels=[3, 7, 11, 15]):
        super(MultiScaleConvAE, self).__init__()
        self.enc1 = MultiScaleConvBlock(n_features, hidden_dim, kernels=kernels)
        self.enc2 = nn.Conv1d(hidden_dim, latent_dim, kernel_size=5, padding=2)
        self.dec1 = nn.Conv1d(latent_dim, hidden_dim, kernel_size=5, padding=2)
        self.dec2 = nn.Conv1d(hidden_dim, n_features, kernel_size=5, padding=2)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, x):
        x_t = x.transpose(1, 2)
        h1 = self.enc1(x_t)
        z = self.relu(self.enc2(h1))
        h2 = self.relu(self.dec1(z))
        out = self.tanh(self.dec2(h2))
        return out.transpose(1, 2)

    def get_latent(self, x):
        h1 = self.enc1(x.transpose(1, 2))
        return self.relu(self.enc2(h1))

# -------------------------------------------------------------------------
# 2. Dilated Temporal Convolutional Network (DilatedTCN_AE)
# -------------------------------------------------------------------------
class TCNBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, dilation=1):
        super(TCNBlock, self).__init__()
        pad = (kernel_size - 1) * dilation // 2
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size, padding=pad, dilation=dilation)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size=kernel_size, padding=pad, dilation=dilation)
        self.res = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.act = nn.ReLU()

    def forward(self, x):
        h = self.act(self.conv1(x))
        h = self.conv2(h)
        return self.act(h + self.res(x))

class DilatedTCN_AE(nn.Module):
    def __init__(self, n_features=1, num_channels=[8, 16], latent_dim=8):
        super(DilatedTCN_AE, self).__init__()
        self.enc_blocks = nn.ModuleList([
            TCNBlock(n_features, num_channels[0], dilation=1),
            TCNBlock(num_channels[0], num_channels[1], dilation=2),
            TCNBlock(num_channels[1], latent_dim, dilation=4)
        ])
        self.dec_blocks = nn.ModuleList([
            TCNBlock(latent_dim, num_channels[1], dilation=4),
            TCNBlock(num_channels[1], num_channels[0], dilation=2),
            TCNBlock(num_channels[0], n_features, dilation=1)
        ])
        self.tanh = nn.Tanh()

    def forward(self, x):
        h = x.transpose(1, 2)
        for b in self.enc_blocks:
            h = b(h)
        for b in self.dec_blocks[:-1]:
            h = b(h)
        out = self.tanh(self.dec_blocks[-1](h))
        return out.transpose(1, 2)

    def get_latent(self, x):
        h = x.transpose(1, 2)
        for b in self.enc_blocks:
            h = b(h)
        return h

# -------------------------------------------------------------------------
# 3. TinyGRU_AE (Recurrent Gated Autoencoder)
# -------------------------------------------------------------------------
class TinyGRU_AE(nn.Module):
    def __init__(self, n_features=1, hidden_dim=16, latent_dim=8, num_layers=1):
        super(TinyGRU_AE, self).__init__()
        self.encoder = nn.GRU(n_features, hidden_dim, num_layers=num_layers, batch_first=True, bidirectional=True)
        self.enc_proj = nn.Linear(hidden_dim * 2, latent_dim)
        self.decoder = nn.GRU(latent_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.dec_proj = nn.Linear(hidden_dim, n_features)
        self.tanh = nn.Tanh()

    def forward(self, x):
        B, W, C = x.shape
        enc_out, _ = self.encoder(x)
        z = F.relu(self.enc_proj(enc_out))
        dec_out, _ = self.decoder(z)
        out = self.tanh(self.dec_proj(dec_out))
        return out

    def get_latent(self, x):
        enc_out, _ = self.encoder(x)
        return self.enc_proj(enc_out)

# -------------------------------------------------------------------------
# 4. LightTransformerAE (Compact 1-Layer Micro-Attention)
# -------------------------------------------------------------------------
class LightTransformerAE(nn.Module):
    def __init__(self, n_features=1, d_model=16, n_heads=2, dim_feedforward=32):
        super(LightTransformerAE, self).__init__()
        self.in_proj = nn.Linear(n_features, d_model)
        self.norm = nn.LayerNorm(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=dim_feedforward, batch_first=True, dropout=0.0, layer_norm_eps=1e-5
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.out_proj = nn.Linear(d_model, n_features)
        self.tanh = nn.Tanh()

    def forward(self, x):
        h = self.norm(self.in_proj(x))
        z = self.transformer(h)
        out = self.tanh(self.out_proj(z))
        return out

    def get_latent(self, x):
        h = self.norm(self.in_proj(x))
        return self.transformer(h)

# -------------------------------------------------------------------------
# 5. HybridPredictiveAE (Dual-Head: Reconstruction + Forecasting)
# -------------------------------------------------------------------------
class HybridPredictiveAE(nn.Module):
    """Joint reconstruction of input window + prediction of future segment."""
    def __init__(self, n_features=1, hidden_dim=16, latent_dim=8, pred_len=5):
        super(HybridPredictiveAE, self).__init__()
        self.pred_len = pred_len
        self.enc = nn.Sequential(
            nn.Conv1d(n_features, hidden_dim, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, latent_dim, kernel_size=5, padding=2),
            nn.ReLU()
        )
        self.dec_recon = nn.Sequential(
            nn.Conv1d(latent_dim, hidden_dim, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, n_features, kernel_size=5, padding=2),
            nn.Tanh()
        )
        self.pred_head = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, pred_len * n_features),
            nn.Tanh()
        )

    def forward(self, x):
        x_t = x.transpose(1, 2)
        z = self.enc(x_t)
        recon = self.dec_recon(z).transpose(1, 2)
        
        # Predict next segment from temporal summary
        z_summary = torch.mean(z, dim=2)
        pred = self.pred_head(z_summary).view(x.size(0), self.pred_len, -1)
        return recon, pred

    def get_latent(self, x):
        return self.enc(x.transpose(1, 2))
