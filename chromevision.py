import torch
import cv2
import numpy as np
import chrome_utils.model_utils as model_utils
import chrome_utils.cut_utils as cut_utils
import chrome_utils.transforms as transforms

class Chromevision():
    def __init__(self, model, cutter):
        self.model = model
        self.cutterModel = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True, trust_repo=True)
        self.device = "cuda" if torch.cuda.is_available() else "cpu" # Have to define the device for later use
        #Does not currently use a specific cutter because it would be be
        #Harder to read and by default worse performance      

    def identify(self, image_path):
        encoder = self.model.encoder_query
        self.model.eval() #We swap to the mode we need (evaluation)
        encoder.eval()

        image = cv2.imread(image_path)
        cuts = self.cutterModel(image)

        for cut in cuts.xyxy[0].data:     # Goes through the different cuts corner values           
            x1, y1, x2, y2 = cut[:4]
            # Adding the crop to the boxes list
            imageCut = image[int(y1):int(y2),int(x1):int(x2)].copy()
            
            imageCut = cut_utils.scaleCuts(imageCut)
            if imageCut is None:
                continue
            imageCut = transforms.evalTransform(imageCut).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                features = encoder(imageCut)
                probabilities = torch.softmax(features, dim=1)

            predicted_class = np.argmax(probabilities.cpu().numpy())

            startPoint_rectangle = (int(x1), int(y1)) #We add the box and class to the original image
            endPoint_rectangle = (int(x2), int(y2))
            color_bgr = (0, 255, 0)
            thickness = 2

            cv2.rectangle(image, startPoint_rectangle, endPoint_rectangle, color_bgr, thickness)                     

            org = (int(x1), int(y1) - 10)
            color_bgr = (0, 0, 255)
            frontScale = 0.5            
            cv2.putText(image, str(predicted_class), org, cv2.FONT_HERSHEY_SIMPLEX, frontScale, color_bgr, thickness)
        return image