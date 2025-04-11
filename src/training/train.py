import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import time
import logging
import os
from typing import Dict, Any, Tuple
from ..model import autoencoder, losses # Import relative to src

logger = logging.getLogger(__name__)

def train_epoch(model: nn.Module, dataloader: DataLoader, loss_fn, optimizer: optim.Optimizer, device: torch.device) -> float:
    """Runs one training epoch."""
    model.train() # Set model to training mode
    total_loss = 0.0
    num_batches = 0
    for batch_idx, (x_batch, y_batch) in enumerate(dataloader):
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(x_batch) # mu, theta (or mu, theta, pi)

        # Calculate loss
        if model.distribution == 'ZINB':
            y_batch = y_batch.round().long()
            loss = loss_fn(y_batch, outputs[0], outputs[1], outputs[2])
        else: # NB
            y_batch = y_batch.round().long()
            loss = loss_fn(y_batch, outputs[0], outputs[1])

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    return avg_loss

def validate_epoch(model: nn.Module, dataloader: DataLoader, loss_fn, device: torch.device) -> float:
    """Runs one validation epoch."""
    model.eval() # Set model to evaluation mode
    total_loss = 0.0
    num_batches = 0
    with torch.no_grad(): # Disable gradient calculations
        for x_batch, y_batch in dataloader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)

            # Forward pass
            outputs = model(x_batch)

            # Calculate loss
            if model.distribution == 'ZINB':
                y_batch = y_batch.round().long()
                loss = loss_fn(y_batch, outputs[0], outputs[1], outputs[2])
            else: # NB
                y_batch = y_batch.round().long()
                loss = loss_fn(y_batch, outputs[0], outputs[1])

            total_loss += loss.item()
            num_batches += 1

    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    return avg_loss


def train_model(X_train: np.ndarray, Y_train: np.ndarray,
                X_val: np.ndarray, Y_val: np.ndarray,
                model_config: Dict[str, Any],
                training_config: Dict[str, Any],
                save_path: str) -> Tuple[nn.Module, Dict]:
    """
    Main function to train the autoencoder model.

    Args:
        X_train, Y_train: Training data (scaled input, raw counts target).
        X_val, Y_val: Validation data.
        model_config: Dictionary with model architecture parameters.
        training_config: Dictionary with training hyperparameters.
        save_path: Path to save the best model checkpoint.

    Returns:
        Tuple containing the trained model and training history.
    """
    logger.info("Starting model training...")
    start_time = time.time()

    # --- Setup Device ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # --- Create Datasets and DataLoaders ---
    logger.info("Creating PyTorch Datasets and DataLoaders...")
    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(Y_train))
    val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(Y_val))

    batch_size = training_config['batch_size']
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True if device=='cuda' else False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True if device=='cuda' else False)
    logger.info(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # --- Initialize Model ---
    logger.info("Initializing model...")
    input_dim = X_train.shape[1]
    output_dim = Y_train.shape[1] # Number of genes
    model = autoencoder.CountAutoencoder(
        input_dim=input_dim,
        output_dim=output_dim,
        encoder_layer_dims=model_config['encoder_layers'],
        bottleneck_dim=model_config['bottleneck_size'],
        decoder_layer_dims=model_config['decoder_layers'],
        distribution=model_config['distribution'],
        activation_fn=nn.ReLU() if model_config['activation'] == 'relu' else nn.SELU(), # Example mapping
        dropout_rate=training_config.get('dropout_rate', 0.0)
    ).to(device)
    logger.info(f"Model:\n{model}")

    # --- Define Loss and Optimizer ---
    if model.distribution == 'NB':
        loss_fn = losses.negative_binomial_loss_torch
    elif model.distribution == 'ZINB':
        loss_fn = losses.zinb_loss_torch
    else:
        raise ValueError(f"Unsupported distribution: {model.distribution}")

    if training_config['optimizer'].lower() == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=training_config['learning_rate'])
    elif training_config['optimizer'].lower() == 'adamw':
         optimizer = optim.AdamW(model.parameters(), lr=training_config['learning_rate'])
    else:
        # Add other optimizers if needed
        raise ValueError(f"Unsupported optimizer: {training_config['optimizer']}")
    logger.info(f"Optimizer: {type(optimizer).__name__}, LR: {training_config['learning_rate']}")

    # Learning rate scheduler (optional)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5, verbose=True)

    # --- Training Loop ---
    logger.info("Starting training loop...")
    epochs = training_config['epochs']
    patience = training_config['early_stopping_patience']
    best_val_loss = float('inf')
    epochs_no_improve = 0
    history = {'train_loss': [], 'val_loss': []}

    os.makedirs(os.path.dirname(save_path), exist_ok=True) # Ensure save directory exists

    for epoch in range(epochs):
        epoch_start_time = time.time()

        train_loss = train_epoch(model, train_loader, loss_fn, optimizer, device)
        val_loss = validate_epoch(model, val_loader, loss_fn, device)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)

        epoch_duration = time.time() - epoch_start_time
        logger.info(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Time: {epoch_duration:.2f}s")

        # Update LR scheduler
        scheduler.step(val_loss)

        # Checkpoint and Early Stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            # Save the best model state
            torch.save(model.state_dict(), save_path)
            logger.info(f"Validation loss improved. Saved best model to {save_path}")
        else:
            epochs_no_improve += 1
            logger.info(f"Validation loss did not improve for {epochs_no_improve} epoch(s).")

        if epochs_no_improve >= patience:
            logger.info(f"Early stopping triggered after {epoch+1} epochs.")
            break

    # --- Load Best Model ---
    logger.info(f"Loading best model weights from {save_path}")
    model.load_state_dict(torch.load(save_path, map_location=device))

    total_training_time = time.time() - start_time
    logger.info(f"Training finished in {total_training_time:.2f} seconds.")

    return model, history

