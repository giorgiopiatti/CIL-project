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
from model import Model

RANDOM_STATE = 42
BATCH_SIZE = 256
DATA_DIR = '../data'


MAX_EPOCHS = 25
import os
from dotenv import load_dotenv
load_dotenv()
BASE_DIR_RESULTS = os.getenv('BASE_DIR_RESULTS')
DIR_RESULTS=BASE_DIR_RESULTS+'res_optuna/'
EXPERIMENT_NAME = 'NCF'
EXPERIMENT_NAME+='-'+str(uuid.uuid4())[:8]
N_OPTUNA_TRIALS = 100
DEBUG = False
NUM_GPUS = 6

NUM_WORKERS_LOADER = 4

os.makedirs(DIR_RESULTS+EXPERIMENT_NAME)

def create_trial_params(trial):
    params = {
        "emb_size_user": trial.suggest_int('emb_size_user', 2,64),
        "emb_size_movie": trial.suggest_int('emb_size_movie', 2,64),
        "p_dropout": trial.suggest_float('p_dropout', 0.0, 0.5),
        "lr": trial.suggest_float('lr', 1e-5, 1e-2), 
        "weight_decay": trial.suggest_float('lr', 1e-5, 1e-2), 
    }
    return params


def trial_fn(trial):
    params = create_trial_params(trial)
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



import dask
if __name__ == "__main__":
    from optuna_run_dask import run_optuna
    dask.config.set({"distributed.worker.daemon":False} )

    best_params = run_optuna(trial_fn=trial_fn, experiment_name=EXPERIMENT_NAME, n_trials=N_OPTUNA_TRIALS, n_gpus=NUM_GPUS)
    #from optuna_single_gpu import run_optuna
    #best_params = run_optuna(trial_fn=trial_fn, experiment_name=EXPERIMENT_NAME, n_trials=N_OPTUNA_TRIALS)
    

    best_params = {
        "emb_size_user": 16,
        "emb_size_movie": 16,
        "p_dropout": 0.0,
        "lr": best_params['lr'], 
        "weight_decay": 1e-4,
    }

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


    model = Model(**best_params)
    checkpoint_callback = pl.callbacks.ModelCheckpoint(dirpath=f'{DIR_RESULTS}/{EXPERIMENT_NAME}/checkpoints-{EXPERIMENT_NAME}_best', save_top_k=1, monitor=f'val_rmse')
    
    proxies = {
    'http': 'http://proxy.ethz.ch:3128',
    'https': 'http://proxy.ethz.ch:3128',
    }
    neptune_logger = NeptuneLogger(
        project="TiCinesi/CIL-project", 
        
        mode = 'debug' if DEBUG else 'async',
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
    neptune_logger.experiment['results/end_model'].upload(File(f'{EXPERIMENT_NAME}-predictedSubmission.csv'))