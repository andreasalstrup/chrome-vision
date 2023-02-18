import torch
import cv2
import numpy as np

image = cv2.imread('data/E45Vejle_1011.jpg')

image = cv2.resize(image, (840, 840))

model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True, trust_repo=True)

results = model(image)

count = 0
print("\ntensor([x1, y1, x2, y2, score, class_id])\n")

for result in results.xyxy[0]:
    x1, y1, x2, y2, score, class_id = result
    print(result)

    startPoint_rectangle = (int(x1), int(y1))
    endPoint_rectangle = (int(x2), int(y2))
    color_bgr = (0, 255, 0)
    thickness = 2

    cv2.rectangle(image, startPoint_rectangle, endPoint_rectangle, color_bgr, thickness)
    
    text = f'{model.names[int(class_id)]} {score:.2f}'
    org = (int(x1), int(y1) - 10)
    color_bgr = (255, 0, 0)
    frontScale = 0.5

    cv2.putText(image, text, org, cv2.FONT_HERSHEY_SIMPLEX, frontScale, color_bgr, thickness)

    count += 1

print(f'\nObjects detected: {count}\n')

cv2.imshow('Result', image)
cv2.waitKey(0)
cv2.destroyAllWindows()