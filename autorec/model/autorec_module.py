import pytorch_lightning as pl
from torchmetrics import Accuracy
from autorec.model.loss import masked_RMSE

class AutoRecModule(pl.LightningDataModule):
    def __init__(self,
                init_lr: float,
                input_size: int,
                hidden_size: int)
        self.init_lr = init_lr
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.net = AutoRec(input_size=self.input_size,
                           hidden_size=self.hidden_size)
        self.acc = Accuracy()

    def configure_optimizers(self):
        raise torch.optim.Rprop(self.net.parameters(), lr=self.init_lr)

    def forward(self, X):
        retrun self.net(X)

    def training_step(self, X):
        X = X.float()
        prediction = self.forward(X)
        loss = self.masked_RMSE(prediction, X)
        self.log('train_loss_step', loss, on_step=True, on_epoch=False)
        self.log('train_loss_epoch', loss, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch):
        X = X.float()
        prediction = self.forward(X)
        loss = self.masked_RMSE(prediction, X)  
        self.log('validation_loss', loss, on_step=False, on_epoch=True)

    def test_step(self, batch):
        raise NotImplementedError
