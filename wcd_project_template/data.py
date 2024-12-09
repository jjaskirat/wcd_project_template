from abc import ABC, abstractmethod
import numpy as np
import os
from typing import Tuple, Union, Optional, Iterable

import albumentations as A
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd

from wcd_project_template.utils import load_from_import_str

class Data(ABC, Dataset):
    """
    This is the Base class for any Dataset created in the project.
    It inherits from the PyTorch Dataset.

    It implements the __getitem__ and the __len__ functions.
    There are 3 other functions that need to be implemented.
    The get_input, get_label and apply_transformation methods.
    (Look into the methods for more details)

    Sample Usage:
    dataset = Data(...)
    data[1] -> returns the input and labels as a Tuple
    """
    def __init__(
        self,
        df,
        config_data: dict,
        transform: Optional[A.Compose] = None
    ):
        """initializes the Dataset

        Args:
            df (pd.core.frame.DataFrame): the DataFrame containing the paths of the images and labels
            config_data (dict): config used to execute the dataset and dataloader
            transform (Optional[A.Compose], optional): Transforms that need to be applied to the dataset. Defaults to None.
        """
        self.config_data = config_data
        self.df = df
        self.root_dir = self.config_data['root_dir']
        self.transform = transform

    def __getitem__(
        self,
        idx: int
    ) -> Tuple[np.ndarray, int, torch.Tensor]:
        """This method gets the idx element from the dataset

        Args:
            idx (int): the idx of the data point to be retrieved

        Returns:
            Tuple[np.ndarray, int, torch.Tensor]: input and labels
        """
        row = self.df.iloc[idx]
        input = self.get_input(row)
        label = self.get_label(row)
        if self.transform is not None:
            input, label = self.apply_transform(input, label)
        return input, label
    
    def __len__(self) -> int:
        """returns the length of the dataset

        Returns:
            int: length
        """
        return len(self.df)
    
    def get_dataloader(self) -> torch.utils.data.DataLoader:
        """returns the dataloader created on the dataset

        Returns:
            torch.utils.data.DataLoader: dataloader
        """
        dataloader = DataLoader(self, **self.config_data['dataloader'])
        return dataloader

    @abstractmethod
    def get_input(
        self,
        *args,
        **kwargs
    ) -> np.ndarray:
        """gets the input from the df row

        Raises:
            NotImplementedError: to be implemented in the child class

        Returns:
            np.ndarray: the image
        """
        raise NotImplementedError
    
    @abstractmethod
    def get_label(
        self,
        *args,
        **kwargs
    ) -> Union[np.ndarray, int]:
        """gets the label from the df row

        Raises:
            NotImplementedError: to be implemented in the child class

        Returns:
            Union[np.ndarray, int]: the label
        """
        raise NotImplementedError
    
    @abstractmethod
    def apply_transform(
        self,
        *args,
        **kwargs
    ) -> Tuple[np.ndarray, int, torch.Tensor]:
        """applies the transformation on both the image and the label respectively

        Raises:
            NotImplementedError: to be implemented in the child class

        Returns:
            Tuple[np.ndarray, int, torch.Tensor]: the transformed images and labels
        """
        raise NotImplementedError