import pytorch_lightning as pl


class AutoRecModule(pl.LightningDataModule):
    def __init__(self,
                init_lr: float,
                input_size: int,
                hidden_size: int)
        raise NotImplementedError

    def forward(self, X):
        raise NotImplementedError

    def training_step(self, batch):
        raise NotImplementedError

    def validation_step(self, batch):
        raise NotImplementedError

    def test_step(self, batch):
        raise NotImplementedError

    def configure_optimizers(self):
        raise NotImplementedError
