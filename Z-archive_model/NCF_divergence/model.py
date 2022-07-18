import torch 
import torch.nn as nn 
import pytorch_lightning as pl
from zmq import device
number_of_users, number_of_movies = (10000, 1000)

"""
Base model

"""
class Model(pl.LightningModule):

    def __init__(self, emb_size_user, emb_size_movie, p_dropout, lr, hidden_size_1, hidden_size_2, weight_decay,
            loss_coeff_0,
            loss_coeff_1,
            loss_coeff_2,
            loss_coeff_3,
            loss_coeff_4
            ):
        super().__init__()
        self.save_hyperparameters()
        self.user_embedding = nn.Embedding(number_of_users, emb_size_user)
        self.movie_embedding = nn.Embedding(number_of_movies, emb_size_movie)
        
        self.user_bias = nn.Embedding(number_of_users, 1)
        self.movie_bias = nn.Embedding(number_of_movies, 1)

        self.ncf = nn.Sequential(
            nn.Linear(emb_size_user+emb_size_movie, hidden_size_1),
            nn.ReLU(),
            nn.Dropout(p_dropout),
            nn.Linear(hidden_size_1, hidden_size_2),
            nn.ReLU(),
            nn.Dropout(p_dropout),
            nn.Linear(hidden_size_2, 1)
        )
        self.emb_size_user = emb_size_user
        self.lr = lr
        self.weight_decay = weight_decay


        device='cuda' if torch.cuda.is_available() else 'cpu'

        self.coeff = torch.tensor([loss_coeff_0, loss_coeff_1, loss_coeff_2, loss_coeff_3,loss_coeff_4], device=device)
        self.coeff = self.coeff / torch.sum(self.coeff)
        

    def forward(self, batch):
        users, movies = batch[:, 0], batch[:, 1]

        users_embedding = self.user_embedding(users)
        movies_embedding = self.movie_embedding(movies)
     
        input = torch.cat([users_embedding, movies_embedding], dim=1)
        out =  self.ncf(input) + self.user_bias(users) + self.movie_bias(movies)
        return torch.nn.functional.sigmoid(out)*4+1.0
    

    def loss(self,yhat, y):
        return torch.sum((y*torch.log(y/yhat)+yhat-y)*self.coeff[y-1])

    def rmse_metric(self, yhat, y):
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
        rmse = self.rmse_metric(yhat, y)
        self.log('train_rmse', rmse, on_epoch=True, on_step=True, prog_bar=True)
        self.log('train_loss', loss, on_epoch=True, on_step=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x,y = batch
        yhat = self(x)
        loss = self.loss(yhat, y)
        rmse = self.rmse_metric(yhat, y)
        self.log('val_rmse', rmse, on_epoch=True, on_step=True, prog_bar=True)
        self.log('val_loss', loss, on_epoch=True, on_step=True, prog_bar=True)
        return loss
