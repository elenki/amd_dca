import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import time
import logging
import os
from typing import Dict, Any, Tuple

from amd_dca.model import autoencoder, losses

logger = logging.getLogger(__name__)

def train_epoch(model: nn.Module, dataloader: DataLoader, loss_fn, optimizer: optim.Optimizer, device: torch.device) -> float:
    model.train()
    total_loss = 0.0
    for x_batch, y_batch in dataloader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        outputs = model(x_batch)
        if model.distribution == 'ZINB':
            loss = loss_fn(y_batch, outputs[0], outputs[1], outputs[2])
        else:
            loss = loss_fn(y_batch, outputs[0], outputs[1])
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

def validate_epoch(model: nn.Module, dataloader: DataLoader, loss_fn, device: torch.device) -> float:
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for x_batch, y_batch in dataloader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            outputs = model(x_batch)
            if model.distribution == 'ZINB':
                loss = loss_fn(y_batch, outputs[0], outputs[1], outputs[2])
            else:
                loss = loss_fn(y_batch, outputs[0], outputs[1])
            total_loss += loss.item()
    return total_loss / len(dataloader)

def train_model(
    X_train: np.ndarray, Y_train: np.ndarray,
    X_val: np.ndarray,   Y_val: np.ndarray,
    model_config: Dict[str, Any],
    training_config: Dict[str, Any],
    save_path: str
) -> Tuple[nn.Module, Dict]:
    logger.info("Starting model training...")
    start_time = time.time()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(Y_train))
    val_dataset   = TensorDataset(torch.FloatTensor(X_val),   torch.FloatTensor(Y_val))

    batch_size = training_config['batch_size']
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4,
                              pin_memory=(device.type=='cuda'))
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, num_workers=4,
                              pin_memory=(device.type=='cuda'))
    logger.info(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # Initialize model
    model = autoencoder.CountAutoencoder(
        input_dim=X_train.shape[1],
        encoder_layers=model_config['encoder_layers'],
        bottleneck_dim=model_config['bottleneck_size'],
        decoder_layers=model_config['decoder_layers'],
        output_dim=Y_train.shape[1],
        distribution=model_config['distribution'],
        activation=nn.ReLU() if model_config['activation']=='relu' else nn.SELU(),
        dropout=training_config.get('dropout_rate', 0.0)
    ).to(device)
    logger.info(f"Model initialized on {device}")

    # Loss & optimizer
    if model.distribution == 'NB':
        loss_fn = losses.negative_binomial_loss
    else:
        loss_fn = losses.zinb_loss

    opt_name = training_config['optimizer'].lower()
    if opt_name == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=training_config['learning_rate'])
    elif opt_name == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=training_config['learning_rate'])
    else:
        raise ValueError(f"Unsupported optimizer: {opt_name}")

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min',
        patience=training_config.get('lr_patience', 5),
        factor=0.5, verbose=True
    )

    # Training loop
    best_val_loss = float('inf')
    epochs_no_improve = 0
    history = {'train_loss': [], 'val_loss': []}
    patience = training_config['early_stopping_patience']

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    for epoch in range(training_config['epochs']):
        train_loss = train_epoch(model, train_loader, loss_fn, optimizer, device)
        val_loss   = validate_epoch(model, val_loader,   loss_fn, device)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        scheduler.step(val_loss)

        logger.info(f"Epoch {epoch+1}/{training_config['epochs']} — "
                    f"Train: {train_loss:.4f}, Val: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), save_path)
            logger.info(f"Validation improved; saved checkpoint to {save_path}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                logger.info("Early stopping triggered.")
                break

    # load best
    model.load_state_dict(torch.load(save_path, map_location=device))
    total_time = time.time() - start_time
    logger.info(f"Training complete in {total_time:.1f}s")
    return model, history