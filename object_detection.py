import torch
import cv2
import numpy as np

image = cv2.imread('data/E45Vejle_1011.jpg')

image = cv2.resize(image, (840, 840))

model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True, trust_repo=True)

results = model(image)

count = 0
print("\ntensor([x1, y1, x2, y2, score, class_id])\n")
boxes = list()
for result in results.xyxy[0]:
    x1, y1, x2, y2, score, class_id = result    

    ####### Adding the crop to the boxes list
    croppedImage = image[int(y1):int(y2)-1,int(x1):int(x2)].copy()
    boxes.append(croppedImage)

    count += 1


print(f'\nObjects detected: {count}\n')

for elem in boxes:    
    cv2.imshow('Result', elem)
    cv2.waitKey(0)
cv2.destroyAllWindows()
