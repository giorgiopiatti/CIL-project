import torch 
import torch.nn as nn 
import pytorch_lightning as pl
number_of_users, number_of_movies = (10000, 1000)

"""
Base model

"""
class Model(pl.LightningModule):

    def __init__(self, emb_size_user, emb_size_movie, p_dropout, lr, weight_decay=0):
        super().__init__()
        self.save_hyperparameters()
        self.user_embedding = nn.Embedding(number_of_users, emb_size_user)
        self.movie_embedding = nn.Embedding(number_of_movies, emb_size_movie)
        
        self.ncf = nn.Sequential(
            nn.Linear(emb_size_user+emb_size_movie, 64),
            nn.ReLU(),
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.ReLU()
        )
        self.emb_size_user = emb_size_user
        self.lr = lr
        self.weight_decay = weight_decay
        

    def forward(self, batch):
        users, movies = batch[:, 0], batch[:, 1]

        users_embedding = self.user_embedding(users)
        movies_embedding = self.movie_embedding(movies)
     
        input = torch.cat([users_embedding, movies_embedding], dim=1)
        return self.ncf(input)
    
    def loss(self, yhat, y):
        """
        RMSE
        """
        return torch.sqrt(torch.mean((yhat -y)**2))

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), self.lr, weight_decay=self.weight_decay)
    
    def training_step(self, batch, batch_idx):
        x,y = batch
        yhat = self(x)
        loss = self.loss(yhat, y)
        self.log('train_rmse', loss, on_epoch=True, on_step=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x,y = batch
        yhat = self(x)
        loss = self.loss(yhat, y)
        self.log('val_rmse', loss, on_epoch=True, on_step=True, prog_bar=True)
        return loss
