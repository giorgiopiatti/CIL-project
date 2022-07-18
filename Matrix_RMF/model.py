import torch 
import torch.nn as nn 
import pytorch_lightning as pl
number_of_users, number_of_movies = (10000, 1000)

"""
Base model

"""
class Model(pl.LightningModule):

    def __init__(self, emb_size, alpha, lr):
        super().__init__()
        self.save_hyperparameters()

        self.lr = lr
        self.U = nn.parameter.Parameter(torch.rand((number_of_users, emb_size)))
        self.V = nn.parameter.Parameter(torch.rand((number_of_movies, emb_size)))
        self.alpha = alpha

    def forward(self, batch):
        users, movies = batch[:, 0], batch[:, 1]

        mul = torch.mul(self.U[users], self.V[movies])

        return torch.sum(mul, dim = 1)
    
    def loss(self, yhat, y):
        """
        RMSE
        """
        return self.alpha/2*(torch.norm(self.U)**2 + torch.norm(self.V)**2) + torch.sum((y-yhat)**2)

    def rmse_metric(self, yhat, y):
        """
        RMSE
        """
        return torch.sqrt(torch.mean((yhat -y)**2))

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), self.lr)
    
    def training_step(self, batch, batch_idx):
        x,y = batch
        yhat = self(x)
        loss = self.loss(yhat, y)
        rmse = self.rmse_metric(yhat,y)
        self.log('train_rmse', rmse, on_epoch=True, on_step=True, prog_bar=True)
        self.log('train_loss', loss, on_epoch=True, on_step=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x,y = batch
        yhat = self(x)
        loss = self.loss(yhat, y)
        rmse = self.rmse_metric(y, yhat)
        self.log('val_rmse', rmse, on_epoch=True, on_step=True, prog_bar=True)
        self.log('val_loss', loss, on_epoch=True, on_step=True, prog_bar=True)
        return loss
