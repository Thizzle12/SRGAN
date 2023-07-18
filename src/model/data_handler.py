import os

import cv2
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T
from torchvision.io import ImageReadMode, read_image


class DatasetHandler(Dataset):
    def __init__(
        self,
        files_path: str,
        low_res: tuple[int, int] | int = 28,
        preprocess: T.Compose = None,
        scale_factor: int = 4,
        *args,
        **kwargs,
    ) -> None:
        super().__init__()
        self.files_path = files_path
        self.files = os.listdir(files_path)

        self.low_res_resize = T.Resize(size=(low_res, low_res))
        self.high_res_resize = T.Resize(
            size=(low_res * scale_factor, low_res * scale_factor)
        )
        # self.preprocess = preprocess  #
        # self.preprocess = T.Compose([T.Normalize(0.5, std=0.5)])
        self.preprocess = T.Compose([T.ToTensor()])

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index):
        img = cv2.imread(os.path.join(self.files_path, self.files[index]))
        if img.shape[-1] > 1:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # if self.preprocess:
        img = self.preprocess(img)

        target = self.high_res_resize(img)
        img = self.low_res_resize(img)

        return img / 255, target / 255
        # return img, target
