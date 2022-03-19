import pandas as pd
from typing import Optional
from torch.utils.data import Dataset
import os


class MovieLensDataset(Dataset):

    def __init__(self,
                data_dir: str,
                device: str='cuda:0',
                model_type: str='item'):


        self.device = device
        self.model_type = model_type
        self.data = None

        # set data dirs
        rating_path = os.path.join(data_dir, 'ratings.dat')

        # read rating file
        rating_cols = ['user_id', 'movie_id', 'rating', 'timestamp']
        self.ratings = pd.read_table(rating_path,
                              sep='::',
                              names=rating_cols,
                              engine='python',
                              encoding='ISO-8859-1')

        if  self.model_type == 'item':
            self.data = pd.pivot_table(self.ratings, index=['movie_id'], values=['rating'], columns='user_id', fill_value=0)
        else:
            self.data = pd.pivot_table(self.ratings, index=['user_id'], values=['rating'], columns='movie_id', fill_value=0)

    def __getitem__(self, idx):
        return self.data['idx']

    def __len__(self):
        return len(self.data)
    

def main():
    data_dir = '/disk/teang1995/ml-10M100K'
    movielens_dataset = MovieLensDataset(data_dir=data_dir,
                                         device='cuda:0',
                                         model_type='item')
    print(type(movielens_dataset.data))
    print(len(movielens_dataset.data))

if __name__ == "__main__":
    main()