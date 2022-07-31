import os
import uuid

import dask
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from neptune.new.types import File
from pytorch_lightning.loggers import NeptuneLogger
from sklearn.model_selection import train_test_split

from dataset import (DatasetComplete, DatasetValidation,
                     extract_matrix_users_movies_ratings, save_predictions)
from model import Model
from optuna_run_dask import run_optuna
from optuna_single_gpu import run_optuna as run_optuna_single_gpu

number_of_users, number_of_movies = (10000, 1000)
RANDOM_STATE = 58
BATCH_SIZE = 256
DATA_DIR = '../data'

MAX_EPOCHS = 40

import os
from dotenv import load_dotenv
load_dotenv()
BASE_DIR_RESULTS = os.getenv('BASE_DIR_RESULTS')
DIR_RESULTS=BASE_DIR_RESULTS+'res_optuna/'
EXPERIMENT_NAME = 'AE_base'
EXPERIMENT_NAME += '-'+str(uuid.uuid4())[:8]
N_OPTUNA_TRIALS = 150
DEBUG = False
NUM_GPUS = 1

os.makedirs(DIR_RESULTS+EXPERIMENT_NAME)


def create_trial_params(trial):
    params = {
        "hidden_size": trial.suggest_int("hidden_size", 16, 256),
        "encoding_size": trial.suggest_int("encoding_size", 16, 512),
        "z_p_dropout": trial.suggest_float('z_p_dropout', 0.0, 1.0),
        "lr": trial.suggest_float('lr', 1e-5, 1e-2),
        "num_layers": trial.suggest_int("num_layers", 2, 6),
        "weight_decay": trial.suggest_float('weight_decay', 1e-5, 1e-2),
    }
    return params


def trial_fn(trial):
    params = create_trial_params(trial)

    # Data source and split into val and train
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

        centered_user = np.where(user == 0, user, (user-mean)/std)
        users_centered.append(centered_user)
        users_mean.append(mean)
        users_std.append(std)

    matrix_users_movies_train = np.stack(users_centered)
    users_mean_train = np.stack(users_mean)
    users_std_train = np.stack(users_std)

    d_train = DatasetComplete(matrix_users_movies_train)
    train_dataloader = torch.utils.data.DataLoader(
        d_train, batch_size=BATCH_SIZE, drop_last=True, shuffle=True)

    matrix_users_movies_val, _ = extract_matrix_users_movies_ratings(val_pd)
    d_val = DatasetValidation(matrix_users_movies_train, matrix_users_movies_val)
    val_dataloader = torch.utils.data.DataLoader(
        d_val, batch_size=len(d_val), drop_last=False, shuffle=False)

    d_test = DatasetComplete(matrix_users_movies_train)
    test_dataloader = torch.utils.data.DataLoader(
        d_test, batch_size=BATCH_SIZE, drop_last=False, shuffle=False)

    model = Model(
        users_mean=users_mean_train,
        users_std=users_std_train, **params)

    checkpoint_callback = pl.callbacks.ModelCheckpoint(dirpath=f'{DIR_RESULTS}/{EXPERIMENT_NAME}/OPTUNA_checkpoints_{trial.number}',
                                                       save_top_k=1, monitor=f'val_rmse')
    trainer = pl.Trainer(
        max_epochs=MAX_EPOCHS,
        accelerator="gpu" if torch.cuda.is_available() else None,
        devices=1,
        log_every_n_steps=1,
        detect_anomaly=True,
        track_grad_norm=2,
        enable_progress_bar=False,
        callbacks=[checkpoint_callback],
        default_root_dir=DIR_RESULTS
    )
    trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

    return checkpoint_callback.state_dict()['best_model_score'].cpu().item()


if __name__ == "__main__":
    if NUM_GPUS == 1:
        best_params = run_optuna_single_gpu(
            trial_fn=trial_fn, experiment_name=EXPERIMENT_NAME, n_trials=N_OPTUNA_TRIALS)
    else:
        dask.config.set({"distributed.worker.daemon": False})
        best_params = run_optuna(trial_fn=trial_fn, experiment_name=EXPERIMENT_NAME,
                                 n_trials=N_OPTUNA_TRIALS, n_gpus=NUM_GPUS)
    # Data source and split into val and train
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

        centered_user = np.where(user == 0, user, (user-mean)/std)
        users_centered.append(centered_user)
        users_mean.append(mean)
        users_std.append(std)

    matrix_users_movies_train = np.stack(users_centered)
    users_mean_train = np.stack(users_mean)
    users_std_train = np.stack(users_std)

    d_train = DatasetComplete(matrix_users_movies_train)
    train_dataloader = torch.utils.data.DataLoader(
        d_train, batch_size=BATCH_SIZE, drop_last=True, shuffle=True)

    matrix_users_movies_val, _ = extract_matrix_users_movies_ratings(val_pd)
    d_val = DatasetValidation(matrix_users_movies_train, matrix_users_movies_val)
    val_dataloader = torch.utils.data.DataLoader(
        d_val, batch_size=len(d_val), drop_last=False, shuffle=False)

    d_test = DatasetComplete(matrix_users_movies_train)
    test_dataloader = torch.utils.data.DataLoader(
        d_test, batch_size=BATCH_SIZE, drop_last=False, shuffle=False)

    model = Model(
        users_mean=users_mean_train,
        users_std=users_std_train, **best_params)

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=f'{DIR_RESULTS}/{EXPERIMENT_NAME}/checkpoints-{EXPERIMENT_NAME}_best', save_top_k=1, monitor=f'val_rmse')

    proxies = {
        'http': 'http://proxy.ethz.ch:3128',
        'https': 'http://proxy.ethz.ch:3128',
    }
    neptune_logger = NeptuneLogger(
        project="TiCinesi/CIL-project",
        api_key='eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJmMzQyZmQ3MS02OGM5LTQ2Y2EtOTEzNC03MjBjMzUyN2UzNDMifQ==',
        mode='debug' if DEBUG else 'async',
        name=EXPERIMENT_NAME,
        tags=[],  # optional
        proxies=proxies,
        source_files='**/*.py'
    )
    trainer = pl.Trainer(
        max_epochs=MAX_EPOCHS,
        accelerator="gpu" if torch.cuda.is_available() else None,
        devices=1,
        log_every_n_steps=1,
        detect_anomaly=True,
        track_grad_norm=2,
        enable_progress_bar=True,
        callbacks=[checkpoint_callback],
        logger=neptune_logger
    )
    trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

    predictions = trainer.predict(model, dataloaders=test_dataloader)
    yhat = torch.concat(predictions)
    save_predictions(f'{EXPERIMENT_NAME}-predictedSubmission.csv', yhat)
    neptune_logger.experiment['results/end_model'].upload(
        File(f'{EXPERIMENT_NAME}-predictedSubmission.csv'))
