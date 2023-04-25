import os
from torch.utils.data import Dataset
from torchvision.io import read_image
import pandas as pd

class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None):        
        self.img_names = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform        

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):        
        img_path = os.path.join(self.img_dir, self.img_names.iloc[idx,0]) #Need to index with 0 because of the way panda works
        image = read_image(img_path)        
        if self.transform:
            image = self.transform(image)
        return image