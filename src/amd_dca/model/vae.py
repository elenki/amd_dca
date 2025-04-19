# src/amd_dca/model/vae.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple
import logging

logger = logging.getLogger(__name__)

class CountVAE(nn.Module):
    """
    Variational Autoencoder for counts (NB or ZINB).
    Encoder → q(z|x) parameters → reparametrize → Decoder → NB/ZINB heads.
    """
    def __init__(
        self,
        input_dim: int,
        encoder_layers: List[int],
        latent_dim: int,
        decoder_layers: List[int],
        output_dim: int,
        distribution: str = "NB",        # "NB" or "ZINB"
        activation: str = "relu",        # "relu", "selu", "leaky_relu", or nn.Module
        dropout: float = 0.0,
    ):
        super().__init__()
        self.distribution = distribution.upper()
        self.latent_dim  = latent_dim
        self.dropout     = dropout

        # 1) map activation string → nn.Module
        if isinstance(activation, str):
            act = activation.lower()
            if act == "relu":
                act_fn = nn.ReLU()
            elif act == "selu":
                act_fn = nn.SELU()
            elif act == "leaky_relu":
                act_fn = nn.LeakyReLU()
            else:
                raise ValueError(f"Unknown activation: {activation}")
        elif isinstance(activation, nn.Module):
            act_fn = activation
        else:
            raise TypeError("activation must be a string or an nn.Module")

        # --- Encoder MLP ---
        enc_modules: List[nn.Module] = []
        last = input_dim
        for h in encoder_layers:
            enc_modules.append(nn.Linear(last, h))
            enc_modules.append(act_fn)
            if dropout > 0:
                enc_modules.append(nn.Dropout(dropout))
            last = h
        self.encoder_net = nn.Sequential(*enc_modules)

        # q(z|x) parameters
        self.fc_mu     = nn.Linear(last, latent_dim)
        self.fc_logvar = nn.Linear(last, latent_dim)

        # --- Decoder MLP ---
        dec_modules: List[nn.Module] = []
        last = latent_dim
        for h in decoder_layers:
            dec_modules.append(nn.Linear(last, h))
            dec_modules.append(act_fn)
            if dropout > 0:
                dec_modules.append(nn.Dropout(dropout))
            last = h
        self.decoder_net = nn.Sequential(*dec_modules)

        # NB / ZINB heads
        self.mean_head = nn.Linear(last, output_dim)
        self.disp_head = nn.Linear(last, output_dim)
        if self.distribution == "ZINB":
            self.pi_head = nn.Linear(last, output_dim)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        # Encode
        h = self.encoder_net(x)
        mu_z     = self.fc_mu(h)
        logvar_z = self.fc_logvar(h)

        # Reparameterize
        z = self.reparameterize(mu_z, logvar_z)

        # Decode
        d = self.decoder_net(z)
        mu    = F.softplus(self.mean_head(d)) + 1e-6
        theta = F.softplus(self.disp_head(d)) + 1e-6

        if self.distribution == "ZINB":
            pi = torch.sigmoid(self.pi_head(d))
            # return (mu, theta, pi, mu_z, logvar_z)
            return mu, theta, pi, mu_z, logvar_z

        # NB‐VAE
        return mu, theta, mu_z, logvar_z