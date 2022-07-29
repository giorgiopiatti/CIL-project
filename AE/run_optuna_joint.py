import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from optuna.integration import PyTorchLightningPruningCallback
from pytorch_lightning.loggers import NeptuneLogger
from neptune.new.types import File
import pytorch_lightning as pl
import uuid
import os 
import numpy as np
import math

from dataset import extract_matrix_users_movies_ratings, DatasetComplete, DatasetValidation, save_predictions
from model import Model
from swa_model import SWAModel 
from pytorch_lightning.callbacks import LearningRateMonitor


RANDOM_STATE = 42
DATA_DIR = '../data'
number_of_users, number_of_movies = (10000, 1000)

DIR_RESULTS = '/cluster/scratch/ncorecco/CIL/res_optuna/'
EXPERIMENT_NAME = 'AE SWA manual optuna'
EXPERIMENT_NAME+='-'+str(uuid.uuid4())[:8]
N_OPTUNA_TRIALS = 200
DEBUG = False
NUM_GPUS = 6

NUM_WORKERS_LOADER = 4

os.makedirs(DIR_RESULTS+EXPERIMENT_NAME)

def create_trial_params(trial):
    params = {
        "hidden_size": trial.suggest_int("base__hidden_size", 4,64),
        "encoding_size": trial.suggest_int("base__encoding_size", 4,64),
        "z_p_dropout": trial.suggest_float('base__z_p_dropout',0.0, 1.0), 
        "lr": trial.suggest_float('base__lr', 1e-5, 1e-2), 
        "num_layers": trial.suggest_int("base__num_layers", 2,4),
        "weight_decay": trial.suggest_float('base__weight_decay', 1e-5, 1e-2),
    }

    
    params_ensemble = {}
    params_ensemble['lr_low'] = trial.suggest_float('ensemble__lr_low', 1e-4, 1e-3)
    params_ensemble['lr_high'] = trial.suggest_float('ensemble__lr_high', params_ensemble['lr_low'] + 1e-4, 2e-3)
    params_ensemble['weight_decay'] = trial.suggest_float('ensemble__weight_decay', 0.0, 1e-2)
    params_ensemble['frequency_step'] = trial.suggest_int('ensemble__frequency_step', 1,5)
    trainer_config = {}
    trainer_config['epochs_base'] = trial.suggest_int('trainer__epochs_base', 1, 20)
    trainer_config['num_ensemble_models'] = trial.suggest_int('trainer__num_ensemble_models', 5, 20)
    trainer_config['batch_size'] = trial.suggest_categorical('trainer__batch_size', [32,64,128,256])
    
    return params, params_ensemble, trainer_config



def trial_fn(trial):
    params, params_ensemble, trainer_config = create_trial_params(trial)
    
    num_steps_epoch = math.ceil(number_of_users / trainer_config['batch_size'])
    #Data source and split into val and train
    data_pd = pd.read_csv(DATA_DIR+'/data_train.csv')
    train_pd, val_pd = train_test_split(data_pd, train_size=0.9, random_state=RANDOM_STATE)


    matrix_users_movies_train, _ = extract_matrix_users_movies_ratings(train_pd)
    users_centered = []
    users_mean = []
    users_std = []
    for i in range(number_of_users):
        user = matrix_users_movies_train[i]
        mean = user[user != 0].mean()
        std = user[user != 0].std()
        
        centered_user = np.where(user==0, user, (user-mean)/std)
        users_centered.append(centered_user)
        users_mean.append(mean)
        users_std.append(std)


    matrix_users_movies_train = np.stack(users_centered)
    users_mean_train = np.stack(users_mean)
    users_std_train = np.stack(users_std)

    d_train = DatasetComplete(matrix_users_movies_train)
    train_dataloader = torch.utils.data.DataLoader(d_train, batch_size=trainer_config['batch_size'], drop_last=True, shuffle=True)

    matrix_users_movies_val, _ = extract_matrix_users_movies_ratings(val_pd)
    d_val= DatasetValidation(matrix_users_movies_train, matrix_users_movies_val)
    val_dataloader = torch.utils.data.DataLoader(d_val, batch_size=trainer_config['batch_size'], drop_last=False, shuffle=False)


    model = Model(
    users_mean=users_mean_train,
    users_std=users_std_train, **params)

    #Base model training
    trainer = pl.Trainer(
            max_epochs=trainer_config['epochs_base'], 
            accelerator="gpu" if torch.cuda.is_available() else None,
            devices=1, 
            log_every_n_steps=1, 
            detect_anomaly=True, 
            track_grad_norm=2,
            enable_progress_bar=False,
            callbacks=[],
            default_root_dir=DIR_RESULTS
            )
    trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

    #SWA
    trainer.save_checkpoint(f'{DIR_RESULTS}/{EXPERIMENT_NAME}/OPTUNA_checkpoints_{trial.number}_base.ckpt')
    model = Model.load_from_checkpoint(checkpoint_path=f'{DIR_RESULTS}/{EXPERIMENT_NAME}/OPTUNA_checkpoints_{trial.number}_base.ckpt', **params)

    swa = SWAModel(model, lr_low=params_ensemble['lr_low'], lr_high=params_ensemble['lr_high'], weight_decay=params_ensemble['weight_decay'], 
        frequency_step=params_ensemble['frequency_step']*num_steps_epoch)

    checkpoint_callback = pl.callbacks.ModelCheckpoint(dirpath=f'{DIR_RESULTS}/{EXPERIMENT_NAME}/OPTUNA_checkpoints_{trial.number}', 
    save_top_k=1, monitor=f'val_ensemble_rmse')
    trainer = pl.Trainer(
            max_epochs=params_ensemble['frequency_step']*trainer_config['num_ensemble_models'], 
            accelerator="gpu" if torch.cuda.is_available() else None,
            devices=1, 
            log_every_n_steps=1, 
            detect_anomaly=True, 
            enable_progress_bar=False,
            default_root_dir=DIR_RESULTS,
            callbacks=[
                LearningRateMonitor(logging_interval='step'), checkpoint_callback]
            )
    trainer.fit(swa, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)


    return checkpoint_callback.state_dict()['current_score'].cpu().item()

def help_extract_optuna_params(params):
    from collections import defaultdict
    output = defaultdict(lambda: {})
    for k, v in params.items():
        prefix, suffix = k.split('__', 1)
        output[prefix][suffix] = v
    return output

import dask
if __name__ == "__main__":
    from optuna_run_dask import run_optuna
    dask.config.set({"distributed.worker.daemon":False} )

    #best_params = run_optuna(trial_fn=trial_fn, experiment_name=EXPERIMENT_NAME, n_trials=N_OPTUNA_TRIALS, n_gpus=NUM_GPUS)
    from optuna_single_gpu import run_optuna
    best_params = run_optuna(trial_fn=trial_fn, experiment_name=EXPERIMENT_NAME, n_trials=N_OPTUNA_TRIALS)
    
    proxies = {
    'http': 'http://proxy.ethz.ch:3128',
    'https': 'http://proxy.ethz.ch:3128',
    }
    neptune_logger = NeptuneLogger(
        project="TiCinesi/CIL-project", 
        api_key='eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJmMzQyZmQ3MS02OGM5LTQ2Y2EtOTEzNC03MjBjMzUyN2UzNDMifQ==',
        mode = 'debug' if DEBUG else 'async',
        name=EXPERIMENT_NAME,
        tags=[],  # optional
        proxies=proxies,
        source_files='**/*.py'
    )
    
    best_params = help_extract_optuna_params(best_params)
    params = best_params['base']
    params_ensemble = best_params['ensemble']
    trainer_config = best_params['trainer']

    num_steps_epoch = math.ceil(number_of_users / trainer_config['batch_size'])
    #Data source and split into val and train
    data_pd = pd.read_csv(DATA_DIR+'/data_train.csv')
    train_pd, val_pd = train_test_split(data_pd, train_size=0.9, random_state=RANDOM_STATE)


    matrix_users_movies_train, _ = extract_matrix_users_movies_ratings(train_pd)
    users_centered = []
    users_mean = []
    users_std = []
    for i in range(number_of_users):
        user = matrix_users_movies_train[i]
        mean = user[user != 0].mean()
        std = user[user != 0].std()
        
        centered_user = np.where(user==0, user, (user-mean)/std)
        users_centered.append(centered_user)
        users_mean.append(mean)
        users_std.append(std)


    matrix_users_movies_train = np.stack(users_centered)
    users_mean_train = np.stack(users_mean)
    users_std_train = np.stack(users_std)

    d_train = DatasetComplete(matrix_users_movies_train)
    train_dataloader = torch.utils.data.DataLoader(d_train, batch_size=trainer_config['batch_size'], drop_last=True, shuffle=True)

    matrix_users_movies_val, _ = extract_matrix_users_movies_ratings(val_pd)
    d_val= DatasetValidation(matrix_users_movies_train, matrix_users_movies_val)
    val_dataloader = torch.utils.data.DataLoader(d_val, batch_size=trainer_config['batch_size'], drop_last=False, shuffle=False)

    d_test= DatasetComplete(matrix_users_movies_train)
    test_dataloader = torch.utils.data.DataLoader(d_test, batch_size=trainer_config['batch_size'], drop_last=False, shuffle=False)


    model = Model(
    users_mean=users_mean_train,
    users_std=users_std_train, **params)

    #Base model training
    trainer = pl.Trainer(
            max_epochs=trainer_config['epochs_base'], 
            accelerator="gpu" if torch.cuda.is_available() else None,
            devices=1, 
            log_every_n_steps=1, 
            detect_anomaly=True, 
            track_grad_norm=2,
            enable_progress_bar=False,
            callbacks=[],
            logger=neptune_logger,
            default_root_dir=DIR_RESULTS
            )
    trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

    #SWA
    trainer.save_checkpoint(f'{DIR_RESULTS}/{EXPERIMENT_NAME}/best_base.ckpt')
    model = Model.load_from_checkpoint(checkpoint_path=f'{DIR_RESULTS}/{EXPERIMENT_NAME}/best_base.ckpt', **params)
    
    #SWA
    swa = SWAModel(model, lr_low=params_ensemble['lr_low'], lr_high=params_ensemble['lr_high'], weight_decay=params_ensemble['weight_decay'], 
        frequency_step=params_ensemble['frequency_step']*num_steps_epoch)

    checkpoint_callback = pl.callbacks.ModelCheckpoint(dirpath=f'{DIR_RESULTS}/{EXPERIMENT_NAME}/checkpoints-{EXPERIMENT_NAME}_best', save_top_k=1, 
    monitor=f'val_ensemble_rmse')
    trainer = pl.Trainer(
            max_epochs=params_ensemble['frequency_step']*trainer_config['num_ensemble_models'], 
            accelerator="gpu" if torch.cuda.is_available() else None,
            devices=1, 
            log_every_n_steps=1, 
            detect_anomaly=True, 
            enable_progress_bar=False,
            logger=neptune_logger,
            default_root_dir=DIR_RESULTS,
            callbacks=[
                LearningRateMonitor(logging_interval='step'), checkpoint_callback]
            )
    trainer.fit(swa, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)


    predictions = trainer.predict(swa, dataloaders=test_dataloader)
    yhat = torch.concat(predictions)

    save_predictions(f'{EXPERIMENT_NAME}-predictedSubmission.csv', yhat)
    neptune_logger.experiment['results/end_model'].upload(File(f'{EXPERIMENT_NAME}-predictedSubmission.csv'))