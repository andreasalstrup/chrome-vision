import torch
import cv2
import numpy as np
import chrome_cut
import utilities.chromeUtils as utilis
import utilities.transforms as transforms
#########
#Test setup
import torchvision.models as models
from torch import device, nn

device = "cuda" if torch.cuda.is_available() else "cpu"

# Data loading
BATCH_SIZE = 256
IMAGE_RESIZE = 64

# Model
OUT_FEATURES = 128
QUEUE_SIZE = 65536
MOMENTUM = 0.9
SOFTMAX_TEMP = 0.07

# Encoder
ENCODER = models.resnet50

# Optimizer
OPTIMIZER = torch.optim.Adam
LEARNING_RATE = 0.001
BETAS = (0.9, 0.999)
EPS = 1e-08
WEIGHT_DECAY = 1e-5

# Loss function
LOSS_FN = nn.CrossEntropyLoss()

# Training loop
EPOCHS = 200

from model.chrome_moco import ChromeMoCo
model = ChromeMoCo(base_encoder=ENCODER,
                  feature_dim=OUT_FEATURES,
                  queue_size=QUEUE_SIZE,
                  momentum=MOMENTUM,
                  softmax_temp=SOFTMAX_TEMP).to(device)

model.load_state_dict(torch.load("model/models/ChromeMoCo_BatchSize256_LR0.001_ImageSize64_Epochs200.pt", map_location=torch.device(device)))


#######

class Chromevision():
    def __init__(self, model, cutter):
        self.model = model
        self.cutterModel = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True, trust_repo=True)
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
            
            #imageCut = utilis.scaleCuts(cut) #urrently does not work, faulty implementation?
            if imageCut.any == None:
                continue
            imageCut = transforms.evalTransform(imageCut).unsqueeze(0).to(device)

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
            color_bgr = (255, 0, 0)
            frontScale = 0.5            
            cv2.putText(image, str(predicted_class), org, cv2.FONT_HERSHEY_SIMPLEX, frontScale, color_bgr, thickness)
        return image

mal = Chromevision(model, chrome_cut.ChromeCut())                
hej = mal.identify("data/E45Vejle_1011.jpg")

cv2.imshow('Result', hej)
cv2.waitKey(0)
cv2.destroyAllWindows()