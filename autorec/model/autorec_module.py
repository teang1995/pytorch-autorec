import pytorch_lightning as pl
from pytorch_lightning import LightningModule
import torch
from torchmetrics import Accuracy

from autorec.model.loss import masked_RMSE
from autorec.model.autorec import AutoRec

class AutoRecModule(LightningModule):
    def __init__(self,
                 hidden_size: int,
                 init_lr: float,
                 input_size: int,
                 optimizer: str):
        super().__init__()
        self.init_lr = init_lr
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.optimizer = optimizer
        self.net = AutoRec(input_size=self.input_size,
                           hidden_size=self.hidden_size)


    def configure_optimizers(self):
        if self.optimizer == 'adam':
            optimizer = torch.optim.Adam(self.net.parameters(), lr=self.init_lr)
        elif self.optimizer == 'rmsprop':
            optimizer = torch.optim.RMSprop(self.net.parameters(), lr=self.init_lr)
        else:
            raise NotImplementedError

        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=optimizer,
            gamma=0.99,
            last_epoch=-1,
            verbose=True
        )
        return [optimizer], [scheduler]
    def forward(self, X):
        return self.net(X)

    def training_step(self, X, batch_idx: int):
        X = X.float()
        prediction = self.forward(X)
        loss = masked_RMSE(prediction, X)
        self.log('train_loss_step', loss, on_step=True, on_epoch=False)
        self.log('train_loss_epoch', loss, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, X, batch_idx: int):
        X = X.float()
        prediction = self.forward(X)
        loss = masked_RMSE(prediction, X)  
        self.log('validation_loss', loss, on_step=False, on_epoch=True)

    def test_step(self, batch):
        raise NotImplementedError
