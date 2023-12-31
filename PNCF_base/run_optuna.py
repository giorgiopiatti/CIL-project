import os
import uuid

import dask
import pandas as pd
import pytorch_lightning as pl
import torch
from neptune.new.types import File
from pytorch_lightning.loggers import NeptuneLogger
from sklearn.model_selection import train_test_split

from dataset import (TripletDataset, extract_users_movies_ratings_lists,
                     save_predictions)
from model import NCFDistribution as Model
from optuna_run_dask import run_optuna
from optuna_single_gpu import run_optuna as run_optuna_single_gpu

RANDOM_STATE = 42
BATCH_SIZE = 256
DATA_DIR = '../data'


MAX_EPOCHS = 20
import os
from dotenv import load_dotenv
load_dotenv()
BASE_DIR_RESULTS = os.getenv('BASE_DIR_RESULTS')
DIR_RESULTS=BASE_DIR_RESULTS+'res_optuna/'
EXPERIMENT_NAME = 'PNCF_base'
EXPERIMENT_NAME += '-'+str(uuid.uuid4())[:8]
N_OPTUNA_TRIALS = 250
NUM_WORKERS_LOADER = 4
NEPTUNE_LOG_OFFLINE = True
NUM_GPUS = 1

os.makedirs(DIR_RESULTS+EXPERIMENT_NAME)


def create_trial_params(trial):
    params = {
        "embedding_size": trial.suggest_int('embedding_size', 2, 128),
        "hidden_size": trial.suggest_int('hidden_size', 4, 256),
        "alpha": trial.suggest_float('alpha', 1e-3, 1.0),
        "sigma_prior": trial.suggest_float('sigma_prior', 1e-3, 2.0),
        "distance_0_to_3": trial.suggest_float('distance_0_to_3', 0.0, 1.5),
        "distance_3_to_2": trial.suggest_float('distance_3_to_2', 1e-1, 1.5),
        "distance_2_to_1": trial.suggest_float('distance_2_to_1', 1e-1, 1.5),
        "distance_0_to_4": trial.suggest_float('distance_0_to_4', 1e-1, 1.5),
        "distance_4_to_5": trial.suggest_float('distance_4_to_5', 1e-1, 3.0),
        "p_dropout": trial.suggest_float('p_dropout', 0.1, 0.5),
        "weight_decay": trial.suggest_float('weight_decay', 0.0, 1e-2)
    }
    distance_max = max(params['distance_0_to_3'] + params['distance_3_to_2'] +
                       params['distance_2_to_1'], params['distance_0_to_4'] + params['distance_4_to_5'])
    params['scaling'] = trial.suggest_float('scaling', distance_max, distance_max*2)
    return params


def trial_fn(trial):
    params = create_trial_params(trial)
    model = Model(lr=1e-3, **params)

    # Data source and split into val and train
    data_pd = pd.read_csv(DATA_DIR+'/data_train.csv')
    train_pd, val_pd = train_test_split(data_pd, train_size=0.9, random_state=RANDOM_STATE)

    users_train, movies_train, ratings_train = extract_users_movies_ratings_lists(train_pd)
    d_train = TripletDataset(users_train, movies_train, ratings_train)
    train_dataloader = torch.utils.data.DataLoader(
        d_train, batch_size=BATCH_SIZE, drop_last=True, shuffle=True, num_workers=NUM_WORKERS_LOADER)

    users_val, movies_val, ratings_val = extract_users_movies_ratings_lists(val_pd)
    d_val = TripletDataset(users_val, movies_val, ratings_val)
    val_dataloader = torch.utils.data.DataLoader(
        d_val, batch_size=BATCH_SIZE, drop_last=False, shuffle=False, num_workers=NUM_WORKERS_LOADER)

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

    users_train, movies_train, ratings_train = extract_users_movies_ratings_lists(train_pd)
    d_train = TripletDataset(users_train, movies_train, ratings_train)
    train_dataloader = torch.utils.data.DataLoader(
        d_train, batch_size=BATCH_SIZE, drop_last=True, shuffle=True, num_workers=NUM_WORKERS_LOADER)

    users_val, movies_val, ratings_val = extract_users_movies_ratings_lists(val_pd)
    d_val = TripletDataset(users_val, movies_val, ratings_val)
    val_dataloader = torch.utils.data.DataLoader(
        d_val, batch_size=BATCH_SIZE, drop_last=False, shuffle=False, num_workers=NUM_WORKERS_LOADER)

    test_pd = pd.read_csv(DATA_DIR+'/sampleSubmission.csv')
    users_test, movies_test, ratings_test = extract_users_movies_ratings_lists(test_pd)
    d_test = TripletDataset(users_test, movies_test, ratings_test, is_test_dataset=True)
    test_dataloader = torch.utils.data.DataLoader(
        d_test, batch_size=BATCH_SIZE, drop_last=False, shuffle=False, num_workers=NUM_WORKERS_LOADER)

    model = Model(lr=1e-3, **best_params)
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=f'{DIR_RESULTS}/{EXPERIMENT_NAME}/checkpoints-{EXPERIMENT_NAME}_best', save_top_k=1, monitor=f'val_rmse')

    proxies = {
        'http': 'http://proxy.ethz.ch:3128',
        'https': 'http://proxy.ethz.ch:3128',
    }
    neptune_logger = NeptuneLogger(
        project="TiCinesi/CIL-project",
        
        mode='debug' if NEPTUNE_LOG_OFFLINE else 'async',
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

    predictions = trainer.predict(model, dataloaders=test_dataloader,  ckpt_path='best')
    yhat = torch.concat(predictions)

    save_predictions(f'{EXPERIMENT_NAME}-predictedSubmission.csv', yhat)
    neptune_logger.experiment['results/end_model'].upload(
        File(f'{EXPERIMENT_NAME}-predictedSubmission.csv'))
