import pandas as pd
from typing import Optional
from torch.utils.data import Dataset
import os


class MovieLensDataset(Dataset):

    def __init__(self,
                data_dir: str,
                device: str='cuda:0',
                model_type: str):

        self.data_type = data_type
        self.model_type = model_type
        self.data = None

        movie_path = os.path.join(data_dir, 'movies.dat')
        rating_path = os.path.join(data_dir, 'ratings.dat')
        user_path = ps.path.join(data_dir, 'users.dat')

        movie_cols = ['movie_id', 'title', 'genres']
        movies = pd.read_table(movie_path, 
                               sep='::',
                               header=None,
                               names=movie_cols,
                               engine='python',
                               encoding='ISO-8859-1')

        rating_cols = ['user_id', 'movie_id', 'rating', 'timestamp']
        ratings = pd.read_table(rating_path,
                              sep='::',
                              names=rating_cols,
                              engine='python',
                              encoding='ISO-8859-1')

        user_cols = ['user_id', 'gender', 'age', 'occupation', 'zip']
        users = pd.read_table(user_path,
                              sep='::',
                              names=rating_cols,
                              engine='python',
                              encoding='ISO-8859-1')

        if  self.model_type = 'user':
            self.data = self.build_items
        else:
            self.data = self.build_users

    @staticmethod
    def build_items(items: pd.DataFrame,
                    ratings: pd.DataFrame,
                    users: pd.DataFrame):
        # TODO : i-autorec dataset build 깔끔 구현
        # TODO : movielens-1M, movielens-10M 데이터 불러오는 게 다르다면 어떻게 할 지 결정해야 함.
        raise NotImplementedError
        
    @staticmethod
    def build_users(items: pd.DataFrame,
                    ratings: pd.DataFrame,
                    users: pd.DataFrame):
        # TODO : u-autorec dataset build 깔끔 구현
        raise NotImplementedError

    def __getitem__(self, idx):
        return self.data['idx']

    def __len__(self):
        # return len(self.data)
        return len(self.data)
    

def main():
    # TODO: write test code
    raise NotImplementedError

if __name__ == "__main__":
    main()