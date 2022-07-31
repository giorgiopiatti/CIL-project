import pytorch_lightning as pl
import torch
from torch.optim.swa_utils import AveragedModel

number_of_users, number_of_movies = (10000, 1000)


class SWAModel(pl.LightningModule):

    def __init__(self, base_model, lr_low, lr_high, weight_decay, frequency_step):
        super(SWAModel, self).__init__()
        self.base_model = base_model
        self.lr_low = lr_low
        self.lr_high = lr_high
        self.weight_decay = weight_decay
        self.automatic_optimization = False
        self.average_model = AveragedModel(base_model, device='cpu')

        self.frequency_step = frequency_step
        self.base_model.log = self.log

    def forward(self, batch):
        return NotImplementedError("not to be used")

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.base_model.parameters(), lr=self.lr_low,
                               weight_decay=self.weight_decay)
        return {
            "optimizer": opt,
            "lr_scheduler": torch.optim.lr_scheduler.CyclicLR(opt,
                                                              base_lr=self.lr_low, max_lr=self.lr_high, step_size_up=self.frequency_step//2, step_size_down=self.frequency_step//2, cycle_momentum=False)
        }

    def loss_masked(self, yhat, y, movie_mask):

        diff = (yhat - y) ** 2
        return torch.mean(torch.masked_select(diff, movie_mask))

    def loss_full(self, yhat, y):
        diff = (yhat - y) ** 2
        return torch.mean(diff)

    def training_step(self, batch, batch_idx):

        opt = self.optimizers()
        scheduler = self.lr_schedulers()

        x, _ = batch
        xhat = self.base_model(x)
        movie_mask = x.type(torch.bool)
        loss = self.loss_masked(x, xhat, movie_mask)
        opt.zero_grad()
        self.manual_backward(loss)
        opt.step()

        # Compute fixedpoint loss
        xhat = xhat.detach()
        xhat_hat = self.base_model(xhat)
        loss_fixedpoint = self.loss_full(xhat, xhat_hat)
        opt.zero_grad()
        self.manual_backward(loss_fixedpoint)
        opt.step()

        self.log('train_mse_reconstruction', loss,
                 on_epoch=True, on_step=True, prog_bar=True)
        self.log('train_mse_fixedpoint', loss_fixedpoint,
                 on_epoch=True, on_step=True, prog_bar=True)

        scheduler.step()

        if (self.trainer.global_step//2 + 1) % self.frequency_step == 0:
            self.average_model.update_parameters(self.base_model)

    def rmse_metric(self, yhat, y, movie_mask):
        diff = (yhat - y) ** 2
        return torch.sqrt(torch.mean(torch.masked_select(diff, movie_mask)))

    def validation_step(self, batch, batch_idx):
        self.base_model.validation_step(batch, batch_idx)

        x, y, userd_idx = batch
        xhat = self.average_model(x)

        a = torch.mul(xhat.transpose(0, 1), self.base_model.users_std[userd_idx])

        xhat = a + self.base_model.users_mean[userd_idx]
        xhat = xhat.transpose(0, 1)
        movie_mask = y.type(torch.bool)

        rmse = self.rmse_metric(y, xhat, movie_mask)
        self.log('val_ensemble_rmse', rmse, on_epoch=True, on_step=True, prog_bar=True)

    def predict_step(self, batch, batch_idx):
        x, userd_idx = batch

        xhat = self.average_model(x)

        xhat = torch.mul(xhat.transpose(0, 1), self.base_model.users_std.to(device=xhat.device)[userd_idx.to(
            device=xhat.device)]) + self.base_model.users_mean.to(device=xhat.device)[userd_idx.to(device=xhat.device)]
        xhat = xhat.transpose(0, 1)
        return xhat
