import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import time, os, logging
from typing import Dict, Any, Tuple

from amd_dca.model.vae import CountVAE
from amd_dca.model.losses import negative_binomial_loss, zinb_loss

logger = logging.getLogger(__name__)

def train_epoch_vae(model: nn.Module, loader: DataLoader, loss_fn, kl_weight: float,
                    optimizer: optim.Optimizer, device: torch.device) -> float:
    model.train()
    total_loss = 0.0
    for x_batch, y_batch in loader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()

        out = model(x_batch)
        if model.distribution == "ZINB":
            mu, theta, pi, mu_z, logvar_z = out
            recon = loss_fn(y_batch, mu, theta, pi)
        else:
            mu, theta, mu_z, logvar_z = out
            recon = loss_fn(y_batch, mu, theta)

        # KL divergence: 0.5 * sum(1 + logvar - mu^2 - exp(logvar))
        kl = -0.5 * torch.sum(1 + logvar_z - mu_z.pow(2) - logvar_z.exp(), dim=1)
        kl = torch.mean(kl)

        loss = recon + kl_weight * kl
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)

def validate_epoch_vae(model: nn.Module, loader: DataLoader, loss_fn, kl_weight: float,
                       device: torch.device) -> float:
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            out = model(x_batch)
            if model.distribution == "ZINB":
                mu, theta, pi, mu_z, logvar_z = out
                recon = loss_fn(y_batch, mu, theta, pi)
            else:
                mu, theta, mu_z, logvar_z = out
                recon = loss_fn(y_batch, mu, theta)

            kl = -0.5 * torch.sum(1 + logvar_z - mu_z.pow(2) - logvar_z.exp(), dim=1)
            kl = torch.mean(kl)

            total_loss += (recon + kl_weight * kl).item()

    return total_loss / len(loader)

def train_vae(
    X_train: np.ndarray, Y_train: np.ndarray,
    X_val:   np.ndarray, Y_val:   np.ndarray,
    model_cfg: Dict[str, Any],
    training_cfg: Dict[str, Any],
    save_path: str
) -> Tuple[nn.Module, Dict[str, list]]:
    """
    Train a NB‐ or ZINB‐VAE.
    """
    start = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    # DataLoaders
    bs = training_cfg["batch_size"]
    tr_ds = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(Y_train))
    va_ds = TensorDataset(torch.FloatTensor(X_val),   torch.FloatTensor(Y_val))
    tr_ld = DataLoader(tr_ds, batch_size=bs, shuffle=True,  num_workers=4, pin_memory=device.type=="cuda")
    va_ld = DataLoader(va_ds, batch_size=bs, shuffle=False, num_workers=4, pin_memory=device.type=="cuda")

    # Model
    model = CountVAE(
        input_dim       = X_train.shape[1],
        encoder_layers  = model_cfg["encoder_layers"],
        latent_dim      = model_cfg["latent_dim"],
        decoder_layers  = model_cfg["decoder_layers"],
        output_dim      = Y_train.shape[1],
        distribution    = model_cfg["distribution"],
        activation_fn   = nn.ReLU() if model_cfg["activation"]=="relu" else nn.SELU(),
        dropout_rate    = training_cfg.get("dropout_rate",0.0),
    ).to(device)
    logger.info("VAE model:\n%s", model)

    # Loss / Optimizer
    if model.distribution == "ZINB":
        recon_fn = zinb_loss
    else:
        recon_fn = negative_binomial_loss

    opt_name = training_cfg["optimizer"].lower()
    if opt_name == "adam":
        optimizer = optim.Adam(model.parameters(), lr=training_cfg["learning_rate"])
    elif opt_name == "adamw":
        optimizer = optim.AdamW(model.parameters(), lr=training_cfg["learning_rate"])
    else:
        raise ValueError(f"Unsupported optimizer: {opt_name}")

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
        mode="min", patience=training_cfg.get("lr_patience",5), factor=0.5, verbose=True)

    # Training Loop
    history = {"train_loss": [], "val_loss": []}
    best_val = float("inf")
    wait = 0
    kl_w = model_cfg.get("kl_weight", 1.0)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    for epoch in range(training_cfg["epochs"]):
        tr_loss = train_epoch_vae(model, tr_ld, recon_fn, kl_w, optimizer, device)
        va_loss = validate_epoch_vae(model, va_ld, recon_fn, kl_w, device)
        history["train_loss"].append(tr_loss)
        history["val_loss"].append(va_loss)

        logger.info(f"Epoch {epoch+1}: tr={tr_loss:.4f}, val={va_loss:.4f}")
        scheduler.step(va_loss)

        if va_loss < best_val:
            best_val = va_loss
            wait = 0
            torch.save(model.state_dict(), save_path)
            logger.info("  👉  saved best VAE to %s", save_path)
        else:
            wait += 1
            if wait >= training_cfg["early_stopping_patience"]:
                logger.info("Early stopping at epoch %d", epoch+1)
                break

    # reload best
    model.load_state_dict(torch.load(save_path, map_location=device))
    logger.info("Training done in %.1f sec", time.time()-start)
    return model, history