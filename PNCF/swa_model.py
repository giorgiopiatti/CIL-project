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

    def training_step(self, batch, batch_idx):

        opt = self.optimizers()
        scheduler = self.lr_schedulers()
        loss = self.base_model.training_step(batch, batch_idx)
        opt.zero_grad()
        self.manual_backward(loss)
        opt.step()
        scheduler.step()
        if (self.trainer.global_step + 1) % self.frequency_step == 0:  # End of cycle, need to checkpoint
            self.average_model.update_parameters(self.base_model)

    def rmse_metric(self, yhat, y):
        """
        RMSE
        """
        yhat = yhat.type(torch.float)
        y = y.type(torch.float)
        return torch.sqrt(torch.mean((yhat-y)**2))

    def get_expected(self, pred):
        return 1.0*torch.exp(pred[:, 0]) + 2.0*torch.exp(pred[:, 1]) + 3.0*torch.exp(pred[:, 2]) + 4.0*torch.exp(pred[:, 3]) + 5.0*torch.exp(pred[:, 4])

    def validation_step(self, batch, batch_idx):
        self.base_model.validation_step(batch, batch_idx)

        x, y = batch
        pred, _, _ = self.average_model(x)
        yhat = self.get_expected(pred) - 1
        rmse = self.rmse_metric(y, yhat)
        self.log('val_ensemble_rmse', rmse, on_epoch=True, on_step=True, prog_bar=True)

    def predict_step(self, batch, batch_idx):
        pred, mu, sigma = self.average_model(batch)
        return self.get_expected(pred)
