import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple

class CountAutoencoder(nn.Module):
    """
    A deep count autoencoder for RNA-seq denoising.
    Supports NB or ZINB output distributions.
    """
    def __init__(
        self,
        input_dim: int,
        encoder_layers: List[int],
        bottleneck_dim: int,
        decoder_layers: List[int],
        output_dim: int,
        distribution: str = "NB",
        activation: str = "relu",
        dropout: float = 0.0,
    ):
        super().__init__()
        self.distribution = distribution.upper()
        act = activation

        # --- encoder ---
        dims = [input_dim] + encoder_layers + [bottleneck_dim]
        enc_modules = []
        for i in range(len(dims)-1):
            enc_modules.append(nn.Linear(dims[i], dims[i+1]))
            enc_modules.append(act)
            if dropout>0:
                enc_modules.append(nn.Dropout(dropout))
        self.encoder = nn.Sequential(*enc_modules)

        # --- decoder base ---
        dims = [bottleneck_dim] + decoder_layers
        dec_modules = []
        for i in range(len(dims)-1):
            dec_modules.append(nn.Linear(dims[i], dims[i+1]))
            dec_modules.append(act)
            if dropout>0:
                dec_modules.append(nn.Dropout(dropout))
        self.decoder_base = nn.Sequential(*dec_modules)

        # --- output heads ---
        last_dim = decoder_layers[-1] if decoder_layers else bottleneck_dim
        self.mean_head = nn.Linear(last_dim, output_dim)
        self.disp_head = nn.Linear(last_dim, output_dim)
        if self.distribution == "ZINB":
            self.pi_head = nn.Linear(last_dim, output_dim)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        z = self.encoder(x)
        h = self.decoder_base(z)
        mu = F.softplus(self.mean_head(h)) + 1e-6
        theta = F.softplus(self.disp_head(h)) + 1e-6
        if self.distribution == "ZINB":
            pi = torch.sigmoid(self.pi_head(h))
            return mu, theta, pi
        return mu, theta