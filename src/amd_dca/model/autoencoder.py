import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional

class CountAutoencoder(nn.Module):
    """
    Deep Count Autoencoder model using PyTorch.

    Architecture: Encoder -> Bottleneck -> Decoder -> Parameter Prediction Heads
    """
    def __init__(self,
                 input_dim: int,
                 encoder_layer_dims: List[int],
                 bottleneck_dim: int,
                 decoder_layer_dims: List[int],
                 output_dim: int, # Number of genes
                 distribution: str = 'NB', # 'NB' or 'ZINB'
                 activation_fn: nn.Module = nn.ReLU(),
                 dropout_rate: float = 0.0):
        """
        Initializes the CountAutoencoder.

        Args:
            input_dim: Dimensionality of the input (genes + covariates).
            encoder_layer_dims: List of hidden layer sizes for the encoder.
            bottleneck_dim: Size of the bottleneck layer.
            decoder_layer_dims: List of hidden layer sizes for the decoder.
            output_dim: Dimensionality of the output (number of genes).
            distribution: Output distribution ('NB' or 'ZINB').
            activation_fn: Activation function for hidden layers.
            dropout_rate: Dropout probability.
        """
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.distribution = distribution.upper()
        self.dropout_rate = dropout_rate

        # --- Build Encoder ---
        encoder_layers = []
        last_dim = input_dim
        for dim in encoder_layer_dims:
            encoder_layers.append(nn.Linear(last_dim, dim))
            encoder_layers.append(activation_fn)
            if dropout_rate > 0:
                encoder_layers.append(nn.Dropout(dropout_rate))
            last_dim = dim
        encoder_layers.append(nn.Linear(last_dim, bottleneck_dim))
        encoder_layers.append(activation_fn) # Activation for bottleneck? Optional.
        self.encoder = nn.Sequential(*encoder_layers)

        # --- Build Decoder ---
        decoder_layers_list = []
        last_dim = bottleneck_dim
        for dim in decoder_layer_dims:
            decoder_layers_list.append(nn.Linear(last_dim, dim))
            decoder_layers_list.append(activation_fn)
            if dropout_rate > 0:
                decoder_layers_list.append(nn.Dropout(dropout_rate))
            last_dim = dim
        self.decoder_base = nn.Sequential(*decoder_layers_list) # Base layers before output heads

        # --- Output Heads ---
        # Negative Binomial parameters
        self.mean_head = nn.Linear(last_dim, output_dim)
        self.disp_head = nn.Linear(last_dim, output_dim)

        if self.distribution == 'ZINB':
            # Zero-inflation probability parameter
            self.pi_head = nn.Linear(last_dim, output_dim)

        self._initialize_weights()


    def _initialize_weights(self):
        """Initialize weights using Xavier initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """
        Forward pass through the autoencoder.

        Args:
            x: Input tensor (batch_size, input_dim).

        Returns:
            Tuple of tensors representing distribution parameters.
            For NB: (mean, dispersion)
            For ZINB: (mean, dispersion, pi)
        """
        encoded = self.encoder(x)
        decoded_base = self.decoder_base(encoded)

        # Predict parameters - apply final activations here
        # Use softplus for stability and positivity, exp can explode
        mu = F.softplus(self.mean_head(decoded_base)) + 1e-6 # Add epsilon for numerical stability
        # Ensure dispersion is positive and maybe constrain range
        theta = F.softplus(self.disp_head(decoded_base)) + 1e-6

        if self.distribution == 'ZINB':
            pi = torch.sigmoid(self.pi_head(decoded_base)) # Sigmoid for probability (0, 1)
            return mu, theta, pi
        else: # NB
            return mu, theta

