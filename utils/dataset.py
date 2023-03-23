import os
import pandas as pd
import cv2
from torch.utils.data import Dataset
import torch

class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None):        
        img_names = pd.read_csv(annotations_file)
        self.cutList = list()    
        self.transform = transform        

        ##chromecut
        model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True, trust_repo=True)
        for i in range(len(img_names)):
            img_path = os.path.join(img_dir, img_names.iloc[i, 0])
            image = cv2.imread(img_path)
            cuts = model(image)
            for result in cuts.xyxy[0]:                
                x1, y1, x2, y2, score, monka = result    
                # Adding the crop to the boxes list
                croppedImage = image[int(y1):int(y2)-1,int(x1):int(x2)].copy()
                self.cutList.append(croppedImage)

    def __len__(self):
        return len(self.cutList)

    def __getitem__(self, idx):        
        image = self.cutList[idx]    
        if self.transform:
            image = self.transform(image)
        return image.float()