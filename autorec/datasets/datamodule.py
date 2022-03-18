import pandas as pd
from pytorch_lightning import LightningDataModule
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from typing import List, Tuple

from autorec.datasets.dataset import MovieLensDataset


class MovieLensDataModule(LightningDataModule):

    def __init__(self,
                batch_size: int,
                data_root: str,
                data_size: str,
                device: str,
                model_type: str,
                valid_ratio: float) -> None:
        assert data_size in ['1M', '1m', '10M', '10m'], "data_size must be 1M or 10M"
        assert model_type in ['user', 'item'], "model type must be user or item"

        super().__init__()
        self.batch_size = batch_size
        self.data_root = data_root
        self.data_size = '1m' if data_size in ['1M', '1m'] else '10M100K'
        self.device = device
        self.model_type = model_type
        self.valid_ratio = valid_ratio
        

        self.data_dir = data_root + f'/ml-{self.data_size}'

        # TODO : get matrix size from config file
        self.num_users = 0
        self.num_items = 0

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

        train_df, valid_df = train_test_split(self.movielens_dataset.data, test_size=self.valid_ratio)

        self.train_dataset = train_df.to_numpy()
        self.valid_dataset = valid_df.to_numpy()
        self.test_dataset = valid_df.to_numpy() # 일단 valid_df 넣어둠.

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.valid_dataset, batch_size=self.batch_size, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)


if __name__ == "__main__":
    data_module = MovieLensDataModule(batch_size=64,
                                     data_root='/disk/teang1995',
                                     data_size='1M',
                                     device='cpu',
                                     model_type="item",
                                     valid_ratio=0.2)

    data_module.prepare_data()
    print("prepare done")
    data_module.setup()
    print("setup done")

    train_dataloader = data_module.train_dataloader()
    valid_dataloader = data_module.val_dataloader()
    test_dataloader = data_module.test_dataloader()
    print(len(train_dataloader), len(valid_dataloader), len(test_dataloader))