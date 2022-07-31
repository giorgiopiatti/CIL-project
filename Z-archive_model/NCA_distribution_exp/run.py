import pytorch_lightning as pl
import torch
import pandas as pd
from pytorch_lightning.loggers import NeptuneLogger
from neptune.new.types import File
from sklearn.model_selection import train_test_split

from dataset import extract_matrix_users_movies_ratings, MaskedUserDatasetTrain, MaskedUserDatasetVal, DatasetComplete, save_predictions

#Useful constants
number_of_users, number_of_movies = (10000, 1000)
RANDOM_STATE = 58
BATCH_SIZE = number_of_users
DATA_DIR = '../data'

NUM_BATCH_SAMPLES = 128
NUM_MASKED_MOVIES = 4
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



EXPERIMENT_NAME = 'NCA_dist_exp'
DEBUG = False

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

EPOCHS = 20

from model import Model

params = {'emb_size': 43, 'hidden_size_1': 64, 'hidden_size_2': 64, 
        'decoder_in_size': 32,
        'p_dropout_decoder': 0.1,
'num_heads' : 4,
'alpha': 0.30184561739442606, 'sigma_prior': 0.5280354660742225, 
'distance_0_to_3': 0.845472089209157, 
'distance_3_to_2': 1.0123683337747076, 
'distance_2_to_1': 0.12520765022811642, 
'distance_0_to_4': 0.24389896700863054, 
'distance_4_to_5': 1.9232424230681977, 
'weight_decay': 7.594599314482437e-05, 'scaling': 2.5967376547477308}


model = Model(**params, lr=1e-3)

trainer = pl.Trainer(
        max_epochs=EPOCHS, 
        accelerator="gpu" if torch.cuda.is_available() else None,
        devices=1, 
        log_every_n_steps=1, 
        detect_anomaly=True, 
        track_grad_norm=2,
        logger=neptune_logger,
        )
trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

predictions = trainer.predict(model, dataloaders=test_dataloader, ckpt_path='best')

save_predictions(f'{EXPERIMENT_NAME}-predictedSubmission.csv', predictions[0].cpu().numpy())
neptune_logger.experiment['results/end_model'].upload(File(f'{EXPERIMENT_NAME}-predictedSubmission.csv'))