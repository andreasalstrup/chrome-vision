import os
import pandas as pd
import cv2
from torch.utils.data import Dataset
from torchvision.io import read_image
import torch
import csv

def chromeCutter(annotations_file, img_dir, name, typeOfData):
    if os.path.exists(f"data/leftImg8bit/{typeOfData}/cut/{name}"):
        return
    numOfCuts = 0
    if not(os.path.exists(f"data/leftImg8bit/{typeOfData}/cut")):
        os.mkdir(f"data/leftImg8bit/{typeOfData}/cut")
    os.mkdir(f"data/leftImg8bit/{typeOfData}/cut/{name}")
    img_names = pd.read_csv(annotations_file)        
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True, trust_repo=True)
    for i in range(len(img_names)):
        img_path = os.path.join(img_dir, img_names.iloc[i, 0])
        image = cv2.imread(img_path)
        cuts = model(image)
        for result in cuts.xyxy[0]:     #Goes through the different cuts corner values           
            x1, y1, x2, y2 = result[:4]    
            # Adding the crop to the boxes list
            croppedImage = image[int(y1):int(y2)-1,int(x1):int(x2)].copy()
            cv2.imwrite(f"data/leftImg8bit/{typeOfData}/cut/{name}/{name}{numOfCuts}.jpg", croppedImage)
            numOfCuts += 1
    f = open(f'data/leftImg8bit/indices/{typeOfData}Index/{name}.csv', 'w', newline='')
    writer = csv.writer(f)
    for ind in range(numOfCuts):
        writer.writerow([f"{name}{ind}.jpg"])
    f.close
                
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
        return image.float()