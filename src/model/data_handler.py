import os

from torch.utils.data import Dataset
from torchvision import transforms as T
from torchvision.io import read_image


class DatasetHandler(Dataset):
    def __init__(
        self,
        files_path: str,
        low_res: tuple[int, int] | int = 28,
        preprocess: T.Compose = None,
        *args,
        **kwargs,
    ) -> None:
        super().__init__()
        self.files_path = files_path
        self.files = os.listdir(files_path)

        self.low_res_resize = T.Resize(size=low_res)
        self.high_res_resize = T.Resize(size=low_res * 4)
        self.preprocess = preprocess  # if preprocess else T.Compose([T.ToTensor()])

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index):
        img = read_image(os.path.join(self.files_path, self.files[index]))

        print(img.shape)

        if self.preprocess:
            img = self.preprocess(img)

        target = self.high_res_resize(img)
        img = self.low_res_resize(img)

        return img, target
