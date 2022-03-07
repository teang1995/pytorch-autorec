import pandas as pd
from typing import Optional
from torch.utils.data import Dataset


class MovieLensDataset(Dataset):
    def __init__(self,
                data_dir: str,
                device: str,
                data_type: str):
        # TODO : data_type validation -> user, item 중 하나로.
        raise NotImplementedError

    @staticmethod
    def build_items(items: pd.DataFrame,
                    ratings: pd.DataFrame,
                    users: pd.DataFrame):
        # TODO : i-autorec dataset build 깔끔 구현
        raise NotImplementedError
        
    @staticmethod
    def build_users(items: pd.DataFrame,
                    ratings: pd.DataFrame,
                    users: pd.DataFrame):
        # TODO : u-autorec dataset build 깔끔 구현
        raise NotImplementedError

    def __getitem__(self, idx):
        # TODO : getitem에서 함수 받기? 아니면 여기선 return self.data[idx] 만 해주기? 결정해야 함
        raise NotImplementedError

    def __len__(self):
        # return len(self.data)
        raise NotImplementedError
    

def main():
    # TODO: write test code
    raise NotImplementedError

if __name__ == "__main__":
    main()