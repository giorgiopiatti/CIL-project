import torch
import torch.nn as nn
import pytorch_lightning as pl
number_of_users, number_of_movies = (10000, 1000)

"""
Base model

"""


class Model(pl.LightningModule):

    def __init__(self, emb_size, lr):
        super().__init__()
        self.save_hyperparameters()
        self.user_embedding = nn.Embedding(number_of_users, emb_size)
        self.movie_embedding = nn.Embedding(number_of_movies, emb_size)

        self.ncf = nn.Sequential(
            nn.Linear(emb_size*2, 64),
            nn.ReLU(),
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.ReLU()
        )
        self.lr = lr

    def forward(self, batch):
        users, movies = batch[:, 0], batch[:, 1]

        users_embedding = self.user_embedding(users)
        movies_embedding = self.movie_embedding(movies)

        input = torch.cat([users_embedding, movies_embedding], dim=1)
        return self.ncf(input)

    def loss(self, yhat, y):
        """
        MSE
        """
        return torch.mean((yhat - y)**2)

    def rmse_metric(self, yhat, y):
        """
        RMSE
        """
        return torch.sqrt(torch.mean((yhat - y)**2))

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), self.lr)

    def training_step(self, batch, batch_idx):
        x, y = batch
        yhat = self(x)
        loss = self.loss(yhat, y)
        rmse = self.rmse_metric(yhat, y)
        self.log('train_rmse', rmse, on_epoch=True, on_step=True, prog_bar=True)
        self.log('train_mse', loss, on_epoch=True, on_step=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        yhat = self(x)
        loss = self.loss(yhat, y)
        rmse = self.rmse_metric(yhat, y)
        self.log('val_rmse', rmse, on_epoch=True, on_step=True, prog_bar=True)
        self.log('val_mse', loss, on_epoch=True, on_step=True, prog_bar=True)
        return loss
