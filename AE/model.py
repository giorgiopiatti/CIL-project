import pytorch_lightning as pl
import torch
import torch.nn as nn

number_of_users, number_of_movies = (10000, 1000)


"""
Base model

"""


class Model(pl.LightningModule):

    def __init__(self, users_mean, users_std, lr, weight_decay, num_layers, hidden_size, encoding_size, z_p_dropout):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.weight_decay = weight_decay
        self.automatic_optimization = False

        self.users_std = torch.tensor(users_std)
        self.users_mean = torch.tensor(users_mean)

        encoder_layers = []
        encoder_layers.append(nn.Linear(number_of_movies, hidden_size))

        for _ in range(num_layers-2):
            encoder_layers.append(nn.SELU())
            encoder_layers.append(nn.Linear(hidden_size, hidden_size))

        encoder_layers.append(nn.SELU())
        encoder_layers.append(nn.Linear(hidden_size, encoding_size))

        decoder_layers = []

        decoder_layers.append(nn.Linear(encoding_size, hidden_size))
        decoder_layers.append(nn.SELU())
        for _ in range(num_layers-2):
            decoder_layers.append(nn.Linear(hidden_size, hidden_size))
            decoder_layers.append(nn.SELU())
        decoder_layers.append(nn.Linear(hidden_size, number_of_movies))

        self.encoder = nn.Sequential(*encoder_layers)
        self.decoder = nn.Sequential(*decoder_layers)

        self.z_dropout = nn.Dropout(z_p_dropout)

    def forward(self, batch):

        z = self.encoder(batch)
        z = self.z_dropout(z)
        return self.decoder(z)

    def loss_masked(self, yhat, y, movie_mask):

        diff = (yhat - y) ** 2
        return torch.mean(torch.masked_select(diff, movie_mask))

    def loss_full(self, yhat, y):
        diff = (yhat - y) ** 2
        return torch.mean(diff)

    def rmse_metric(self, yhat, y, movie_mask):
        diff = (yhat - y) ** 2
        return torch.sqrt(torch.mean(torch.masked_select(diff, movie_mask)))

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), self.lr,
                               weight_decay=self.weight_decay)
        return opt

    def training_step(self, batch, batch_idx):
        x, _ = batch
        xhat = self(x)
        movie_mask = x.type(torch.bool)
        loss = self.loss_masked(x, xhat, movie_mask)

        opt = self.optimizers()
        opt.zero_grad()
        self.manual_backward(loss)
        opt.step()

        # Compute fixedpoint loss
        xhat = xhat.detach()
        opt = self.optimizers()
        xhat_hat = self(xhat)
        loss_fixedpoint = self.loss_full(xhat, xhat_hat)
        opt.zero_grad()
        self.manual_backward(loss_fixedpoint)
        opt.step()

        self.log('train_mse_reconstruction', loss,
                 on_epoch=True, on_step=True, prog_bar=True)
        self.log('train_mse_fixedpoint', loss_fixedpoint,
                 on_epoch=True, on_step=True, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        x, y, userd_idx = batch

        xhat = self(x)

        self.users_std = self.users_std.to(device=xhat.device)
        self.users_mean = self.users_mean.to(device=xhat.device)

        a = torch.mul(xhat.transpose(0, 1), self.users_std[userd_idx])

        xhat = a + self.users_mean[userd_idx]
        xhat = xhat.transpose(0, 1)
        movie_mask = y.type(torch.bool)
        loss = self.loss_masked(y, xhat, movie_mask)
        rmse = self.rmse_metric(y, xhat, movie_mask)
        self.log('val_mse', loss, on_epoch=True, on_step=True, prog_bar=True)
        self.log('val_rmse', rmse, on_epoch=True, on_step=True, prog_bar=True)

    def predict_step(self, batch, batch_idx):
        x, userd_idx = batch

        xhat = self(x)

        self.users_std = self.users_std.to(device=xhat.device)
        self.users_mean = self.users_mean.to(device=xhat.device)
        xhat = torch.mul(xhat.transpose(
            0, 1), self.users_std[userd_idx]) + self.users_mean[userd_idx]
        xhat = xhat.transpose(0, 1)
        return xhat
