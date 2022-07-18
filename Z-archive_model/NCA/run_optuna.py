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



from dataset import extract_matrix_users_movies_ratings, MaskedUserDatasetTrain, MaskedUserDatasetVal, DatasetComplete, save_predictions
from model import Model

number_of_users, number_of_movies = (10000, 1000)
RANDOM_STATE = 58
BATCH_SIZE = number_of_users
DATA_DIR = '../data'

MAX_EPOCHS = 15
NUM_BATCH_SAMPLES = 32
NUM_MASKED_MOVIES = 1

DIR_RESULTS = '/cluster/scratch/piattigi/CIL/res_optuna/'
EXPERIMENT_NAME = 'NCA'
EXPERIMENT_NAME+='-'+str(uuid.uuid4())[:8]
N_OPTUNA_TRIALS = 100
DEBUG = False
NUM_GPUS = 1

NUM_WORKERS_LOADER = 4

os.makedirs(DIR_RESULTS+EXPERIMENT_NAME)

def create_trial_params(trial):
    params = {
        "emb_size": trial.suggest_int("emb_size", 4,64),
        "hidden_size_q_1": trial.suggest_int("hidden_size_q_1", 2,128),
        "hidden_size_q_2": trial.suggest_int("hidden_size_q_2", 2,128),
        "hidden_size_v_1": trial.suggest_int("hidden_size_v_1", 2,128),
        "hidden_size_v_2": trial.suggest_int("hidden_size_v_2", 2,128),
        "hidden_size_k_1": trial.suggest_int("hidden_size_k_1", 2,128),
        "hidden_size_k_2": trial.suggest_int("hidden_size_k_2", 2,128),
        "lr": trial.suggest_float('lr', 1e-5, 1e-2), 
        "num_heads": trial.suggest_int("num_heads", 1,4),
        "weight_decay": trial.suggest_float('weight_decay', 1e-5, 1e-2),
    }
    return params

def trial_fn(trial):
    params = create_trial_params(trial)
    model = Model(**params)

    #Data source and split into val and train
    data_pd = pd.read_csv(DATA_DIR+'/data_train.csv')
    train_pd, val_pd = train_test_split(data_pd, train_size=0.9, random_state=RANDOM_STATE)


    matrix_users_movies_train, user_movies_idx_train = extract_matrix_users_movies_ratings(train_pd)
    d_train = MaskedUserDatasetTrain(matrix_users_movies_train, user_movies_idx_train, NUM_BATCH_SAMPLES, NUM_MASKED_MOVIES)
    train_dataloader = torch.utils.data.DataLoader(d_train, batch_size=BATCH_SIZE, drop_last=True, shuffle=True)

    matrix_users_movies_val, _ = extract_matrix_users_movies_ratings(val_pd)
    d_val= MaskedUserDatasetVal(matrix_users_movies_train, matrix_users_movies_val)
    val_dataloader = torch.utils.data.DataLoader(d_val, batch_size=BATCH_SIZE, drop_last=False, shuffle=False)



    
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

    #best_params = run_optuna(trial_fn=trial_fn, experiment_name=EXPERIMENT_NAME, n_trials=N_OPTUNA_TRIALS, n_gpus=NUM_GPUS)
    from optuna_single_gpu import run_optuna
    best_params = run_optuna(trial_fn=trial_fn, experiment_name=EXPERIMENT_NAME, n_trials=N_OPTUNA_TRIALS)
    
    #Data source and split into val and train
    data_pd = pd.read_csv(DATA_DIR+'/data_train.csv')
    train_pd, val_pd = train_test_split(data_pd, train_size=0.9, random_state=RANDOM_STATE)


    matrix_users_movies_train, user_movies_idx_train = extract_matrix_users_movies_ratings(train_pd)
    d_train = MaskedUserDatasetTrain(matrix_users_movies_train, user_movies_idx_train, NUM_BATCH_SAMPLES, NUM_MASKED_MOVIES)
    train_dataloader = torch.utils.data.DataLoader(d_train, batch_size=BATCH_SIZE, drop_last=True, shuffle=True)

    matrix_users_movies_val, _ = extract_matrix_users_movies_ratings(val_pd)
    d_val= MaskedUserDatasetVal(matrix_users_movies_train, matrix_users_movies_val)
    val_dataloader = torch.utils.data.DataLoader(d_val, batch_size=BATCH_SIZE, drop_last=False, shuffle=False)


    #test_pd = pd.read_csv(DATA_DIR+'/sampleSubmission.csv')
    #matrix_users_movies_test, _ = extract_matrix_users_movies_ratings(data_pd)
    d_test= DatasetComplete(matrix_users_movies_train)
    test_dataloader = torch.utils.data.DataLoader(d_test, batch_size=BATCH_SIZE, drop_last=False, shuffle=False)



    model = Model(**best_params)
    checkpoint_callback = pl.callbacks.ModelCheckpoint(dirpath=f'{DIR_RESULTS}/{EXPERIMENT_NAME}/checkpoints-{EXPERIMENT_NAME}_best', save_top_k=1, monitor=f'val_rmse')
    
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
    save_predictions(f'{EXPERIMENT_NAME}-predictedSubmission.csv', predictions[0].cpu().numpy())
    neptune_logger.experiment['results/end_model'].upload(File(f'{EXPERIMENT_NAME}-predictedSubmission.csv'))