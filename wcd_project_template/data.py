from abc import ABC, abstractmethod
import numpy as np
import os
from typing import Tuple, Union, Optional

import Albumentations as A
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd

class Data(ABC, Dataset):
    def __init__(
        self,
        config_data: dict,
        transform: Optional[A.Compose] = None
    ):
        self.config_data = config_data
        self.df = self.config_data['df']
        self.root_dir = self.config_data['root_dir']
        self.transform = transform

    def __getitem__(
        self,
        idx: int
    ) -> Tuple[np.ndarray, int, torch.Tensor]:
        row = self.df.iloc[idx]
        input = self.get_input(row)
        label = self.get_label(row)
        if self.transform is not None:
            input, label = self.apply_transform(input, label)
        return input, label
    
    def __len__(self) -> int:
        return len(self.df)
    
    def get_dataloader(self):
        dataloader = DataLoader(self, **self.config_data['dataloader'])
        return dataloader

    @abstractmethod
    def get_input(
        self,
        *args,
        **kwargs
    ) -> np.ndarray:
        raise NotImplementedError
    
    @abstractmethod
    def get_label(
        self,
        *args,
        **kwargs
    ) -> Union[np.ndarray, int]:
       raise NotImplementedError
    
    @abstractmethod
    def apply_transform(
        self,
        *args,
        **kwargs
    ) -> Tuple[np.ndarray, int, torch.Tensor]:
       raise NotImplementedError