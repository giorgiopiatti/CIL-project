import pytorch_lightning as pl
import torch
import pandas as pd
from pytorch_lightning.loggers import NeptuneLogger
from neptune.new.types import File
from sklearn.model_selection import train_test_split

from dataset import extract_users_movies_ratings_lists, TripletDataset, save_predictions

#Useful constants
number_of_users, number_of_movies = (10000, 1000)
RANDOM_STATE = 42
BATCH_SIZE = 256
DATA_DIR = '../data'

#Data source and split into val and train
import random
class SingleMovieSample(torch.utils.data.Sampler):

    def __init__(self, dataset, batch_size, drop_last=True):
        super().__init__(dataset)

        index_for_movies = {}

        for i, triplet in enumerate(dataset):
           
            str_movie = str(triplet[0][1])

            if str_movie not in index_for_movies:
                index_for_movies[str_movie] = []
            
            index_for_movies[str_movie] +=[i]
        
        trimmed = index_for_movies
        if drop_last:
            trimmed = { 
                k: v[:-(len(v) % batch_size)] 
                for k, v in index_for_movies.items()
            }

        trimmed_values = trimmed.values()

        self._indices = [lst[i:i + batch_size] for lst in trimmed_values for i in range(0, len(lst), batch_size) ]
        random.shuffle(self._indices)

    
    def __len__(self):
        return len(self._indices)
    
    def __iter__(self):
        yield from self._indices


data_pd = pd.read_csv(DATA_DIR+'/data_train.csv')
train_pd, val_pd = train_test_split(data_pd, train_size=0.9, random_state=RANDOM_STATE)


users_train, movies_train, ratings_train = extract_users_movies_ratings_lists(train_pd)
d_train = TripletDataset(users_train, movies_train, ratings_train)
#train_dataloader = torch.utils.data.DataLoader(d_train, batch_sampler=SingleMovieSample(d_train, BATCH_SIZE, drop_last=False))
train_dataloader = torch.utils.data.DataLoader(d_train, batch_size=BATCH_SIZE, drop_last=False, shuffle=True)

users_val, movies_val, ratings_val = extract_users_movies_ratings_lists(val_pd)
d_val= TripletDataset(users_val, movies_val, ratings_val)
val_dataloader = torch.utils.data.DataLoader(d_val, batch_size=BATCH_SIZE, drop_last=False, shuffle=False)


test_pd = pd.read_csv(DATA_DIR+'/sampleSubmission.csv')
users_test, movies_test, ratings_test = extract_users_movies_ratings_lists(test_pd)
d_test= TripletDataset(users_test, movies_test, ratings_test, is_test_dataset=True)
test_dataloader = torch.utils.data.DataLoader(d_test, batch_size=BATCH_SIZE, drop_last=False, shuffle=False)



EXPERIMENT_NAME = 'NCF_net_learning'
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

from model import Model
model = Model( 
    emb_size_user=32, emb_size_movie=32, p_dropout=0.5, lr=1e-3, weight_decay=0
    )

trainer = pl.Trainer(
        max_epochs=50, 
        accelerator="gpu" if torch.cuda.is_available() else None,
        devices=1, 
        log_every_n_steps=1, 
        detect_anomaly=True, 
        track_grad_norm=2,
        logger=neptune_logger,
        )

trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

predictions = trainer.predict(model, dataloaders=test_dataloader)

yhat = torch.concat(predictions)

save_predictions(f'{EXPERIMENT_NAME}-predictedSubmission.csv', yhat)
neptune_logger.experiment['results/end_model'].upload(File(f'{EXPERIMENT_NAME}-predictedSubmission.csv'))