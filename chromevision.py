import torch
import cv2
import numpy as np
from chrome_utils.cut_utils import scaleCuts
from torchvision import transforms

transform_eval = transforms.Compose([
    transforms.ToPILImage(),
    #transforms.RandomResizedCrop(224),
    #transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


class Chromevision():
    def __init__(self, model, cutter):
        # Does not currently use a specific cutter because it would be be
        # harder to read and by default worse performance      
        self.model = model
        self.cutterModel = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True, trust_repo=True)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def identify(self, image_path):
        # Get the encoder from the model
        encoder = self.model.encoder_query

        # Set the model to evaluation mode
        self.model.eval()
        encoder.eval()

        # Load and cut the image
        image = cv2.imread(image_path)
        cuts = self.cutterModel(image)

        # Goes through the different cuts corner values
        for cut in cuts.xyxy[0].data:      
            # Get the coordinates of the cut    
            x1, y1, x2, y2 = cut[:4]

            # Adding the crop to the boxes list
            imageCut = image[int(y1):int(y2),int(x1):int(x2)].copy()

            # Resizing the image to 32x32
            imageCut = scaleCuts(imageCut, 32)
            if imageCut is None:
                continue

            # Transforming the image to a tensor
            imageCut = transform_eval(imageCut).unsqueeze(0).to(self.device)
            
            # Process the image through the encoder
            with torch.no_grad():
                features = encoder(imageCut)
                probabilities = torch.softmax(features, dim=1)

            predicted_class = np.argmax(probabilities.cpu().numpy())

            # Draw the rectangle and add class to the image
            startPoint_rectangle = (int(x1), int(y1))
            endPoint_rectangle = (int(x2), int(y2))
            color_bgr = (0, 255, 0)
            thickness = 2
            cv2.rectangle(image, startPoint_rectangle, endPoint_rectangle, color_bgr, thickness)                     

            # Add the class to the image
            org = (int(x1), int(y1) - 10)
            color_bgr = (255, 0, 0)
            frontScale = 0.5            
            cv2.putText(image, str(predicted_class), org, cv2.FONT_HERSHEY_SIMPLEX, frontScale, color_bgr, thickness)

            # Add the probability to the image
            org = (int(x1), int(y1) - 30)
            color_bgr = (255, 0, 0)
            frontScale = 0.5
            cv2.putText(image, str(probabilities.cpu().numpy()[0][predicted_class]), org, cv2.FONT_HERSHEY_SIMPLEX, frontScale, color_bgr, thickness)

        return image