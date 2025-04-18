import torch
import torch.nn.functional as F
from torch.distributions import NegativeBinomial

MIN_THETA = 1e-4
EPS       = 1e-6

def negative_binomial_loss(
    y_true: torch.Tensor,
    mu: torch.Tensor,
    theta: torch.Tensor,
) -> torch.Tensor:
    y = torch.round(y_true)
    mu    = mu.clamp(min=EPS)
    theta = theta.clamp(min=MIN_THETA)
    p     = (mu / (mu + theta)).clamp(EPS, 1-EPS)
    try:
        nb = NegativeBinomial(total_count=theta, probs=p)
        nll = -nb.log_prob(y)
    except Exception as e:
        raise RuntimeError(f"NB loss failed: {e}")
    return nll.sum(dim=-1).mean()

def zinb_loss(
    y_true: torch.Tensor,
    mu: torch.Tensor,
    theta: torch.Tensor,
    pi: torch.Tensor,
) -> torch.Tensor:
    y = torch.round(y_true)
    mu    = mu.clamp(min=EPS)
    theta = theta.clamp(min=MIN_THETA)
    pi    = pi.clamp(EPS, 1-EPS)
    p     = (mu / (mu + theta)).clamp(EPS, 1-EPS)
    nb    = NegativeBinomial(total_count=theta, probs=p)
    log_nb = nb.log_prob(y)
    zero_case = torch.log(pi + (1-pi)*torch.exp(log_nb) + EPS)
    non_zero = torch.log(1-pi + EPS) + log_nb
    nll = -torch.where(y==0, zero_case, non_zero)
    return nll.sum(dim=-1).mean()