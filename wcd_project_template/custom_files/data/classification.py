import os
from typing import Tuple, Union, Optional

import albumentations as A
import cv2
import pandas as pd

from wcd_project_template.data import Data

class DataClassification(Data):
    def __init__(
            self,
            config_data: dict,
            transform: Optional[A.Compose] = None
        ):
        super().__init__(config_data, transform)

    def get_input(self, row):
        image_path = os.path.join(self.root_dir, row['filename'])
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image
    
    def get_label(self, row):
        class_label = row['Labels']
        return class_label
    
    def apply_transform(self, input, label):
        input = self.transform(image=input)['image']
        return input, label