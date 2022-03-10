from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

class MovieLensDataModule(LightningDataModule):
    def __init__(self,
                data_dir: str,
                device: str,
                batch_size: int,
                split_rate: tuple) -> None:
        # TODO: How to split train/test data safely?
        super().__init__()
        self.data_dir = data_dir
        self.device = device
        self.batch_size = batch_size
        self.split_rate = split_rate

    def prepare_data(self) -> None:
        """
        - download
        - tokenize
        - etc ...
        """
        raise NotImplementedError

    def setup(self) -> None:
        """
        - count number of classes
        - build vocabulary
        - perform train/val/test split
        - apply transforms
        - etc...
        """
        raise NotImplementedError

    def train_dataloader(self) -> DataLoader:
        raise NotImplementedError

    def val_dataloader(self) -> DataLoader:
        raise NotImplementedError

    def test_dataloader(self) -> DataLoader:
        raise NotImplementedError