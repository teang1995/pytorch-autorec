from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from autorec.datasets.dataset import MovieLensDataset


class MovieLensDataModule(LightningDataModule):

    def __init__(self,
                batch_size: int,
                data_dir: str,
                device: str,
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
        self.movielens_dataset = MovieLensDataset(data_dir=self.data_dir,
                                                  device=self.device,
                                                  model_type=self.model_type)

        

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