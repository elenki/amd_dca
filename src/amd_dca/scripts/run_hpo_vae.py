# scripts/run_hpo_vae.py (Optuna hyperparam search for NB‐AE)
import optuna
import logging, datetime, os
import numpy as np
from pathlib import Path
from amd_dca.utils.helpers import find_repo_root, load_config, set_seed
from amd_dca.training.train_vae import train_vae

REPO = find_repo_root()
CFG = REPO / 'config.yaml'
LOGDIR = REPO / 'logs'
DATA = REPO / 'data' / 'processed'
RESULTS = REPO / 'results' / 'hpo'


def setup_logging(name):
    LOGDIR.mkdir(exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    lf = LOGDIR / f"{name}_{ts}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(lf), logging.StreamHandler()]
    )


def objective(trial):
    cfg = load_config(CFG)
    set_seed(cfg['random_seed'])
    sp = cfg['hyperparam_search']['vae']
    # Suggest model params
    model_cfg = cfg['model_vae'].copy()
    model_cfg['encoder_layers'] = trial.suggest_categorical('encoder_layers', sp['param_distributions']['encoder_layers'])
    model_cfg['decoder_layers'] = trial.suggest_categorical('decoder_layers', sp['param_distributions']['decoder_layers'])
    model_cfg['latent_dim'] = trial.suggest_categorical('latent_dim', sp['param_distributions']['latent_dim'])
    model_cfg['kl_weight'] = trial.suggest_categorical('kl_weight', sp['param_distributions']['kl_weight'])
    # Suggest training params
    training_cfg = cfg['training'].copy()
    lr_values = list(map(float, sp['param_distributions']['learning_rate']))
    lr_low, lr_high = min(lr_values), max(lr_values)

    training_cfg['learning_rate'] = trial.suggest_float('learning_rate', lr_low, lr_high, log=True)
    training_cfg['batch_size'] = trial.suggest_categorical('batch_size', sp['param_distributions']['batch_size'])
    training_cfg['dropout_rate'] = trial.suggest_categorical('dropout_rate', sp['param_distributions']['dropout_rate'])

    npz = np.load(DATA / 'preprocessed_data.npz')
    X_train, Y_train = npz['X_train'], npz['Y_train']
    X_val, Y_val = npz['X_val'], npz['Y_val']

    save_dir = RESULTS / 'vae'
    os.makedirs(save_dir, exist_ok=True)
    ckpt = save_dir / f"trial_{trial.number}.pt"

    _, history = train_vae(
        X_train, Y_train, X_val, Y_val,
        model_cfg=model_cfg,
        training_cfg=training_cfg,
        save_path=str(ckpt)
    )
    val_loss = history['val_loss'][-1]
    trial.set_user_attr('params', {**model_cfg, **training_cfg})
    return val_loss


def main():
    setup_logging('hpo_vae')
    cfg = load_config(CFG)
    study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=cfg['random_seed']), pruner=optuna.pruners.MedianPruner())
    study.optimize(objective, n_trials=cfg['hyperparam_search']['vae']['n_trials'])

    logging.info(f"Best trial: {study.best_trial.number}")
    logging.info(f"Best value: {study.best_value}")
    logging.info(f"Best params: {study.best_trial.user_attrs['params']}")

    # Save best_params
    out = RESULTS / 'vae_best_params.json'
    import json
    with open(out, 'w') as f:
        json.dump(study.best_trial.user_attrs['params'], f, indent=2)
    logging.info(f"Saved best VAE params to {out}")

if __name__ == '__main__':
    main()