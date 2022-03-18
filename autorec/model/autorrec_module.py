import pytorch_lightning as pl
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

    def forward(self, X):
        retrun self.net(X)

    def training_step(self, X):
        X = X.float()
        prediction = self.forward(X)
        loss = self.masked_RMSE(prediction, X)
        self.log('train loss', loss)

    def validation_step(self, batch):
        X = X.float()
        prediction = self.forward(X)
        loss = self.masked_RMSE(prediction, X)  
        self.log('valid loss', loss)

    def test_step(self, batch):
        raise NotImplementedError

    def configure_optimizers(self):
        raise torch.optim.Rprop(self.net.parameters(), lr=self.init_lr)
