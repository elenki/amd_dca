"""
Loss functions for the Deep Count Autoencoder.

Key change (2025‑04‑16):
    • Always clamp theta ≥ 1e‑4 (was 1e‑6).
    • Clamp probs into (ε, 1‑ε) with ε = 1e‑6 (was 1e‑8).
    • If a numerical issue still occurs, raise a RuntimeError so the
      training loop stops cleanly instead of returning a non‑grad tensor.
"""
from __future__ import annotations
import torch
import torch.nn.functional as F
from torch.distributions import NegativeBinomial
import logging

logger = logging.getLogger(__name__)

MIN_THETA = 1e-4
EPS       = 1e-6


# ---------------------------------------------------------------------------#
#  Negative Binomial                                                        #
# ---------------------------------------------------------------------------#
def negative_binomial_loss_torch(
    y_true: torch.Tensor,
    mu: torch.Tensor,
    theta: torch.Tensor,
) -> torch.Tensor:
    """
    NB negative log‑likelihood averaged over the batch.

    y_true : integer counts  (batch, genes)
    mu     : predicted mean  (batch, genes)  — positive
    theta  : predicted dispersion            — positive
    """
    # 1) numerical safety
    y_true = torch.round(y_true)                      # ensure integers
    mu     = torch.clamp(mu,    min=EPS)
    theta  = torch.clamp(theta, min=MIN_THETA)

    probs = mu / (mu + theta)
    probs = torch.clamp(probs, min=EPS, max=1.0 - EPS)

    # 2) log‑likelihood
    try:
        nb = NegativeBinomial(total_count=theta, probs=probs)
        nll = -nb.log_prob(y_true)                   # (batch, genes)
    except Exception as e:  # any numeric blow‑up should stop training
        logger.error("NB loss numerical error: %s", e)
        raise RuntimeError("NB loss failed – inspect mu/theta/probs for NaNs")

    return torch.mean(torch.sum(nll, dim=-1))        # scalar


# ---------------------------------------------------------------------------#
#  Zero‑Inflated NB                                                         #
# ---------------------------------------------------------------------------#
def zinb_loss_torch(
    y_true: torch.Tensor,
    mu: torch.Tensor,
    theta: torch.Tensor,
    pi: torch.Tensor,
) -> torch.Tensor:
    """ZINB negative log‑likelihood."""
    y_true = torch.round(y_true)
    mu     = torch.clamp(mu,    min=EPS)
    theta  = torch.clamp(theta, min=MIN_THETA)
    pi     = torch.clamp(pi,    min=EPS, max=1.0 - EPS)

    probs = mu / (mu + theta)
    probs = torch.clamp(probs, min=EPS, max=1.0 - EPS)

    nb = NegativeBinomial(total_count=theta, probs=probs)
    nb_logp = nb.log_prob(y_true)

    zero_case     = torch.log(pi + (1.0 - pi) * torch.exp(nb_logp) + EPS)
    non_zero_case = torch.log(1.0 - pi + EPS) + nb_logp
    nll = -torch.where(y_true == 0, zero_case, non_zero_case)

    return torch.mean(torch.sum(nll, dim=-1))