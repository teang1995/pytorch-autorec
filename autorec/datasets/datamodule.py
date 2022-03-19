import os
import pandas as pd
from pytorch_lightning import LightningDataModule
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from typing import List, Tuple, Optional

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
        self.data_size = '1m' if data_size in ['1M', '1m'] else '10m'

        self.device = device
        self.model_type = model_type
        self.valid_ratio = valid_ratio
        
        self._file_name = 'ml-1m' if self.data_size == '1m' else 'ml-10M100k'
        self.data_dir = data_root + f'/{self._file_name}'

        # TODO : get matrix size from config file
        self.num_users = 0
        self.num_items = 0

    def prepare_data(self) -> None:
        """
        - download
        - tokenize
        - etc ...
        """
        # download data
        os.system(f'rm -r {self.data_dir}')
        os.system(f'wget https://files.grouplens.org/datasets/movielens/ml-{self.data_size}.zip')
        os.system(f'unzip ml-{self.data_size}.zip -d {self.data_root}/')
        os.system(f'rm ml-{self.data_size}.zip')

        self.movielens_dataset = MovieLensDataset(data_dir=self.data_dir,
                                                  device=self.device,
                                                  model_type=self.model_type)
        

    def setup(self, stage: Optional[str] = None) -> None:
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