import os
import pandas as pd
import cv2
import utilities.chromeUtils as utilis
from torch.utils.data import Dataset

import torch
import csv

class ChromeCut():
    def CutImagesInFolder(self, annotations_file, img_dir, new_img_dir, name, new_annotations_file_location):
        # Cuts the images in the given folder  with Yolov5.
        # The new images are put in a /cut/{name} folder.
        # If the folder already exists nothing is done.
        #Parameters:
        #   annotations_file (str): The file path of the csv file
        #   img_dir (str): The file path of the directory of the images
        #   name (str): The name to give the new folder and images
        if os.path.exists(f"{new_img_dir}/cut/{name}"):
            return        
        if not(os.path.exists(f"{new_img_dir}/cut")):
            os.mkdir(f"{new_img_dir}/cut")
        os.mkdir(f"{new_img_dir}/cut/{name}")
        model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True, trust_repo=True)
        numOfCuts = 0
        
        img_names = pd.read_csv(annotations_file)        
        for i in range(len(img_names)):
            img_path = os.path.join(img_dir, img_names.iloc[i, 0])
            image = cv2.imread(img_path)
            cuts = model(image)
            for result in cuts.xyxy[0]:    #Goes through the different cuts corner values           
                x1, y1, x2, y2 = result[:4]    
                # Adding the crop to the boxes list
                croppedImage = image[int(y1):int(y2)-1,int(x1):int(x2)].copy()
                croppedImage  = utilis.scaleCuts(croppedImage)
                if croppedImage == None:
                    continue
                cv2.imwrite(f"{new_img_dir}/cut/{name}/{name}{numOfCuts}.jpg", croppedImage)
                numOfCuts += 1
        f = open(f'{new_annotations_file_location}/{name}.csv', 'w', newline='')
        writer = csv.writer(f)
        for ind in range(numOfCuts):
            writer.writerow([f"{name}{ind}.jpg"])
        f.close

    def CutPictureInMemory(self, file_path): 
        # Cuts the given image with the chosen mode (default yolov5).
        # The cuts are returned and not saved.
        #Parameters: 
        #   file_path (str): The file path of the image          
        image = cv2.imread(file_path)
        model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True, trust_repo=True)
        cuts = model(image)
        result = []
        for cut in cuts.xyxy[0]:     #Goes through the different cuts corner values           
            x1, y1, x2, y2 = cut[:4]  
                # Adding the crop to the boxes list
                ##### We ensure the cuts are of a decent size
            height, width, channels = imageCut.shape
            if height * width < 32 * 32: #We remove very small cuts 
                continue
            if height < 64 | width < 64: #We upscale pictures with a side smaller than 64
                scaleFactor = 64 / min(height, width)
                imageCut = cv2.resize(imageCut, (height * scaleFactor, width * scaleFactor))

            if height > 150 | width > 150: #We downscale pictures with a side bigger than 150
                scaleFactor = 150 / max(height, width)
                imageCut = cv2.resize(imageCut, (height * scaleFactor, width * scaleFactor))
            #####
            result += image[int(y1):int(y2)-1,int(x1):int(x2)].copy()            
        return result