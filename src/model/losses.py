import torch
import torch.nn.functional as F
from torch.distributions import NegativeBinomial, Bernoulli, constraints
import logging

logger = logging.getLogger(__name__)

def negative_binomial_loss_torch(y_true: torch.Tensor, mu: torch.Tensor, theta: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Computes the Negative Binomial negative log-likelihood loss using PyTorch distributions.

    Args:
        y_true: Ground truth counts (batch_size, num_genes), expected to be raw counts.
        mu: Predicted mean parameter (batch_size, num_genes), positive.
        theta: Predicted dispersion parameter (batch_size, num_genes), positive.
        eps: Small epsilon for numerical stability.

    Returns:
        torch.Tensor: Scalar loss value (average negative log-likelihood over batch).
    """
    y_true = y_true.round().long()
    if not (mu > 0).all():
        logger.warning("Non-positive values detected in mu, clamping.")
        mu = torch.clamp(mu, min=eps)
    if not (theta > 0).all():
        logger.warning("Non-positive values detected in theta, clamping.")
        theta = torch.clamp(theta, min=eps)

    # Parameterize NB using total_count (theta) and probs (mu / (mu + theta))
    # Ensure probs are within (0, 1) for stability
    probs = mu / (mu + theta)
    probs = torch.clamp(probs, min=eps, max=1 - eps)

    try:
        nb_dist = NegativeBinomial(total_count=theta, probs=probs)
        # Calculate log probability, sum over genes, average over batch
        neg_log_likelihood = -nb_dist.log_prob(y_true)
    except ValueError as e:
        logger.error(f"Error creating NB distribution or calculating log_prob: {e}")
        logger.error(f"Shapes - y_true: {y_true.shape}, mu: {mu.shape}, theta: {theta.shape}")
        logger.error(f"Sample values - mu: {mu.flatten()[:5]}, theta: {theta.flatten()[:5]}, probs: {probs.flatten()[:5]}")
        # Return a large loss or handle appropriately
        return torch.tensor(float('inf'), device=y_true.device)


    # Sum loss over genes for each sample, then average over the batch
    return torch.mean(torch.sum(neg_log_likelihood, dim=-1))


def zinb_loss_torch(y_true: torch.Tensor, mu: torch.Tensor, theta: torch.Tensor, pi: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Computes the Zero-Inflated Negative Binomial negative log-likelihood loss.

    Args:
        y_true: Ground truth counts (batch_size, num_genes), raw counts.
        mu: Predicted mean parameter (batch_size, num_genes), positive.
        theta: Predicted dispersion parameter (batch_size, num_genes), positive.
        pi: Predicted zero-inflation probability (batch_size, num_genes), between 0 and 1.
        eps: Small epsilon for numerical stability.

    Returns:
        torch.Tensor: Scalar loss value.
    """
    if not (mu > 0).all():
        logger.warning("Non-positive values detected in mu, clamping.")
        mu = torch.clamp(mu, min=eps)
    if not (theta > 0).all():
        logger.warning("Non-positive values detected in theta, clamping.")
        theta = torch.clamp(theta, min=eps)
    # pi is output of sigmoid, should be (0, 1), but clamp just in case
    pi = torch.clamp(pi, min=eps, max=1.0 - eps)

    # --- Calculate NB Log Probability ---
    probs = mu / (mu + theta)
    probs = torch.clamp(probs, min=eps, max=1.0 - eps)
    try:
        nb_dist = NegativeBinomial(total_count=theta, probs=probs)
        nb_log_prob = nb_dist.log_prob(y_true)
    except ValueError as e:
        logger.error(f"Error creating NB distribution or calculating log_prob in ZINB: {e}")
        return torch.tensor(float('inf'), device=y_true.device)

    # --- Calculate Zero-Inflation Component ---
    # Bernoulli likelihood for the zero component (log(pi) if y_true==0, log(1-pi) if y_true>0)
    # Use log_softmax for stability? Or direct calculation.
    # log_prob = log(pi) for zero counts
    # log_prob = log(1-pi) + nb_log_prob for non-zero counts

    # Log prob for y_true == 0: log(pi + (1-pi)*NB(0|mu,theta))
    # Log prob for y_true > 0:  log(1-pi) + NB(y|mu,theta)

    # Use torch.where for selection based on y_true == 0
    zero_case = torch.log(pi + (1.0 - pi) * torch.exp(nb_log_prob) + eps) # Add eps inside log
    non_zero_case = torch.log(1.0 - pi + eps) + nb_log_prob

    mask = (y_true == 0)
    neg_log_likelihood = -torch.where(mask, zero_case, non_zero_case)

    # Sum over genes, average over batch
    return torch.mean(torch.sum(neg_log_likelihood, dim=-1))

