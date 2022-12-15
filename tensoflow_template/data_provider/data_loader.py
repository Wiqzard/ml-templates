import os
import numpy as np
import pandas as pd
import os
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')


class MyDataset(Dataset):
    def __init__(self, root_path: str, data_path: str,  flag="train",) -> None:
        super().__init__()

         # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self) -> None:
        """sets self.data..."""
        pass
    
    def __getitem__(self, idx: int):
        pass


    def __len__(self) -> int:
        pass
    
    
    
    