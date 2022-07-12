from pyro import param
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from optuna.integration import PyTorchLightningPruningCallback
from pytorch_lightning.loggers import NeptuneLogger
from neptune.new.types import File
import pytorch_lightning as pl
import uuid
import os 



from dataset import extract_users_movies_ratings_lists, TripletDataset, save_predictions
from model import NCFDistribution as Model
from swa_model import SWAModel 
from pytorch_lightning.callbacks import LearningRateMonitor


RANDOM_STATE = 42
BATCH_SIZE = 256
DATA_DIR = '../data'


DIR_RESULTS = '/cluster/scratch/piattigi/CIL/res_optuna/'
EXPERIMENT_NAME = 'NCF_distribution SWA manual optuna'
EXPERIMENT_NAME+='-'+str(uuid.uuid4())[:8]
N_OPTUNA_TRIALS = 200
DEBUG = False
NUM_GPUS = 6

NUM_WORKERS_LOADER = 4

os.makedirs(DIR_RESULTS+EXPERIMENT_NAME)

def create_trial_params(trial):
    params = {
        "embedding_size": trial.suggest_int('base__embedding_size', 2,128), 
        "hidden_size": trial.suggest_int('base__hidden_size', 4,256),
        "lr": 1e-3, #trial.suggest_float('lr', 1e-5, 1e-3), 
        "alpha":trial.suggest_float('base__alpha', 1e-3, 1.0),
        "sigma_prior":trial.suggest_float('base__sigma_prior', 1e-3, 2.0),
        "distance_0_to_3" : trial.suggest_float('base__distance_0_to_3', 0.0, 1.5),
        "distance_3_to_2" : trial.suggest_float('base__distance_3_to_2', 1e-1, 1.5),
        "distance_2_to_1" : trial.suggest_float('base__distance_2_to_1', 1e-1, 1.5),
        "distance_0_to_4" : trial.suggest_float('base__distance_0_to_4', 1e-1, 1.5),
        "distance_4_to_5" : trial.suggest_float('base__distance_4_to_5', 1e-1, 2.0),
        "p_dropout":trial.suggest_float('base__p_dropout', 0.1,0.5),
        "weight_decay": trial.suggest_float('base__weight_decay', 0.0, 1e-2)
    }
    distance_max = max(params['distance_0_to_3'] + params['distance_3_to_2'] + params['distance_2_to_1'], params['distance_0_to_4'] + params['distance_4_to_5'])
    params['scaling'] = trial.suggest_float('base__scaling', distance_max, distance_max*2)
    
    params_ensemble = {}
    params_ensemble['lr_low'] = trial.suggest_float('ensemble__lr_low', 1e-4, 1e-3)
    params_ensemble['lr_high'] = trial.suggest_float('ensemble__lr_high', params_ensemble['lr_low'] + 1e-4, 2e-3)
    params_ensemble['weight_decay'] = trial.suggest_float('ensemble__weight_decay', 0.0, 1e-2)
    params_ensemble['frequency_step'] = trial.suggest_int('ensemble__frequency_step', 1,5)
    params_ensemble['epochs_base'] = trial.suggest_int('ensemble__epochs_base', 1, 10)
    
    return params, params_ensemble

NUM_STEPS_EPOCH = 4137
def trial_fn(trial):
    params, params_ensemble = create_trial_params(trial)
    model = Model(**params)

    #Data source and split into val and train
    data_pd = pd.read_csv(DATA_DIR+'/data_train.csv')
    train_pd, val_pd = train_test_split(data_pd, train_size=0.9, random_state=RANDOM_STATE)


    users_train, movies_train, ratings_train = extract_users_movies_ratings_lists(train_pd)
    d_train = TripletDataset(users_train, movies_train, ratings_train)
    train_dataloader = torch.utils.data.DataLoader(d_train, batch_size=BATCH_SIZE, drop_last=True, shuffle=True, num_workers=NUM_WORKERS_LOADER)

    users_val, movies_val, ratings_val = extract_users_movies_ratings_lists(val_pd)
    d_val= TripletDataset(users_val, movies_val, ratings_val)
    val_dataloader = torch.utils.data.DataLoader(d_val, batch_size=BATCH_SIZE, drop_last=False, shuffle=False, num_workers=NUM_WORKERS_LOADER)

    #Base model training
    trainer = pl.Trainer(
            max_epochs=params_ensemble['epochs_base'], 
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
        frequency_step=params_ensemble['frequency_step']*NUM_STEPS_EPOCH)

    checkpoint_callback = pl.callbacks.ModelCheckpoint(dirpath=f'{DIR_RESULTS}/{EXPERIMENT_NAME}/OPTUNA_checkpoints_{trial.number}', 
    save_top_k=1, monitor=f'val_ensemble_rmse')
    trainer = pl.Trainer(
            max_epochs=params_ensemble['frequency_step']*15, 
            accelerator="gpu" if torch.cuda.is_available() else None,
            devices=1, 
            log_every_n_steps=1, 
            detect_anomaly=True, 
            enable_progress_bar=False,
            callbacks=[
                LearningRateMonitor(logging_interval='step'), checkpoint_callback]
            )
    trainer.fit(swa, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)


    return checkpoint_callback.state_dict()['best_model_score'].cpu().item()

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

    best_params = run_optuna(trial_fn=trial_fn, experiment_name=EXPERIMENT_NAME, n_trials=N_OPTUNA_TRIALS, n_gpus=NUM_GPUS)
    #from optuna_single_gpu import run_optuna
    #best_params = run_optuna(trial_fn=trial_fn, experiment_name=EXPERIMENT_NAME, n_trials=N_OPTUNA_TRIALS)
    
    #Data source and split into val and train
    data_pd = pd.read_csv(DATA_DIR+'/data_train.csv')
    train_pd, val_pd = train_test_split(data_pd, train_size=0.9, random_state=RANDOM_STATE)

    users_train, movies_train, ratings_train = extract_users_movies_ratings_lists(train_pd)
    d_train = TripletDataset(users_train, movies_train, ratings_train)
    train_dataloader = torch.utils.data.DataLoader(d_train, batch_size=BATCH_SIZE, drop_last=True, shuffle=True, num_workers=NUM_WORKERS_LOADER)

    users_val, movies_val, ratings_val = extract_users_movies_ratings_lists(val_pd)
    d_val= TripletDataset(users_val, movies_val, ratings_val)
    val_dataloader = torch.utils.data.DataLoader(d_val, batch_size=BATCH_SIZE, drop_last=False, shuffle=False, num_workers=NUM_WORKERS_LOADER)

    test_pd = pd.read_csv(DATA_DIR+'/sampleSubmission.csv')
    users_test, movies_test, ratings_test = extract_users_movies_ratings_lists(test_pd)
    d_test= TripletDataset(users_test, movies_test, ratings_test, is_test_dataset=True)
    test_dataloader = torch.utils.data.DataLoader(d_test, batch_size=BATCH_SIZE, drop_last=False, shuffle=False, num_workers=NUM_WORKERS_LOADER)



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

    model = Model(**params)

    #Data source and split into val and train
    data_pd = pd.read_csv(DATA_DIR+'/data_train.csv')
    train_pd, val_pd = train_test_split(data_pd, train_size=0.9, random_state=RANDOM_STATE)


    users_train, movies_train, ratings_train = extract_users_movies_ratings_lists(train_pd)
    d_train = TripletDataset(users_train, movies_train, ratings_train)
    train_dataloader = torch.utils.data.DataLoader(d_train, batch_size=BATCH_SIZE, drop_last=True, shuffle=True, num_workers=NUM_WORKERS_LOADER)

    users_val, movies_val, ratings_val = extract_users_movies_ratings_lists(val_pd)
    d_val= TripletDataset(users_val, movies_val, ratings_val)
    val_dataloader = torch.utils.data.DataLoader(d_val, batch_size=BATCH_SIZE, drop_last=False, shuffle=False, num_workers=NUM_WORKERS_LOADER)

    #Base model training
    trainer = pl.Trainer(
            max_epochs=params_ensemble['epochs_base'], 
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
        frequency_step=params_ensemble['frequency_step']*NUM_STEPS_EPOCH)

    checkpoint_callback = pl.callbacks.ModelCheckpoint(dirpath=f'{DIR_RESULTS}/{EXPERIMENT_NAME}/checkpoints-{EXPERIMENT_NAME}_best', save_top_k=1, 
    monitor=f'val_ensemble_rmse')
    trainer = pl.Trainer(
            max_epochs=params_ensemble['frequency_step']*15, 
            accelerator="gpu" if torch.cuda.is_available() else None,
            devices=1, 
            log_every_n_steps=1, 
            detect_anomaly=True, 
            enable_progress_bar=False,
            logger=neptune_logger,
            callbacks=[
                LearningRateMonitor(logging_interval='step'), checkpoint_callback]
            )
    trainer.fit(swa, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)


    predictions = trainer.predict(swa, dataloaders=test_dataloader,  ckpt_path='best')
    yhat = torch.concat(predictions)

    save_predictions(f'{EXPERIMENT_NAME}-predictedSubmission.csv', yhat)
    neptune_logger.experiment['results/end_model'].upload(File(f'{EXPERIMENT_NAME}-predictedSubmission.csv'))