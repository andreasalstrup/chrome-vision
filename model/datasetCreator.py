import os
import pandas as pd
from torchvision.io import read_image
from torch.utils.data import Dataset

FILEPATH = "../data/" + "leftImg8bit/" + "train/" + "bremen/"

class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_dir = img_dir
        self.img_names = pd.read_csv(img_dir + annotations_file)        
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):        
        img_path = os.path.join(self.img_dir, self.img_names.iloc[idx, 0])
        image = read_image(img_path)       
        if self.transform:
            image = self.transform(image)
        return image

# Example of how to initialize a dataset:
# ourDataset = CustomImageDataset("index.csv",FILEPATH)