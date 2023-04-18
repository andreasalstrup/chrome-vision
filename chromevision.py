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

from torchvision import transforms
from utilities.transforms import ContrastiveTransform

transform_MoCoV1 = ContrastiveTransform(
                        transforms.Compose([
                            transforms.ToPILImage(),
                            transforms.Resize((IMAGE_RESIZE, IMAGE_RESIZE)),
                            transforms.RandomResizedCrop(IMAGE_RESIZE, scale=(0.2, 1.0)), # 224 -> 64 
                            transforms.RandomGrayscale(p=0.2),
                            transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                            ])
                        )
#######

class Chromevision():
    def __init__(self, model, cutter, merger):
        self.model = model
        self.cutter = cutter
        self.merger = merger
        self.cutterModel = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True, trust_repo=True)

    def green(self, image_path):
         image = cv2.imread(image_path)
         return self.model(image)
    
    def chrome(self, image_path):
        cutterModel = self.cutterModel

        image = cv2.imread(image_path)

        cuts = cutterModel(image)
        
        for cut in cuts.xyxy[0]:     # Goes through the different cuts corner values           
            x1, y1, x2, y2 = cut[:4]
            # Adding the crop to the boxes list
            imageCut = image[int(y1):int(y2),int(x1):int(x2)].copy()
            transformedCut = transform_MoCoV1(imageCut)
            
            self.model.train(False)
            queryImage = transformedCut[0].unsqueeze(0)
            keyImage = transformedCut[1].unsqueeze(0)
            # print(keyImage.shape)
            # testing = keyImage.unsqueeze(0)
            # print(testing.shape)
            output, target = model(query_batch_images=queryImage,key_batch_images=keyImage)

            startPoint_rectangle = (int(x1), int(y1))
            endPoint_rectangle = (int(x2), int(y2))
            color_bgr = (0, 255, 0)
            thickness = 2

            cv2.rectangle(image, startPoint_rectangle, endPoint_rectangle, color_bgr, thickness)
            monka = torch.nn.Softmax(1)
            monka2 = monka(output.data)                        

            _, pred = output.topk(5, 1, True)
            pred = pred.t()
            tal = pred[0].item()

            text = ""

            for elem in pred:
                text += str(elem.item())
                text += ", "
            org = (int(x1), int(y1) - 10)
            color_bgr = (255, 0, 0)
            frontScale = 0.5            
            cv2.putText(image, text, org, cv2.FONT_HERSHEY_SIMPLEX, frontScale, color_bgr, thickness)
            # A
        return image

mal = Chromevision(model, None, None)                
hej = mal.chrome("data/test1.png")

cv2.imshow('Result', hej)
cv2.waitKey(0)
cv2.destroyAllWindows()