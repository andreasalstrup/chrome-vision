import torch
import cv2

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


########

class Chromevision():
    def __init__(self, model, cutter, merger):
        self.model = model
        self.cutter = cutter
        self.merger = merger

    def green(self, image_path):
         image = cv2.imread(image_path)
         return self.model(image)
    
    def chrome(self, image_path):
        cutterModel = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True, trust_repo=True)

        image = cv2.imread(image_path)

        cuts = cutterModel(image)
        
        for cut in cuts.xyxy[0]:     # Goes through the different cuts corner values           
                x1, y1, x2, y2 = cut[:4]  
                # Adding the crop to the boxes list
                imageCut = image[int(y1):int(y2)-1,int(x1):int(x2)].copy()
                result = self.model(imageCut)

                # A

mal = Chromevision(model, None, None)                
hej = mal.green("data/E45Vejle_1011.jpg")
print(hej)