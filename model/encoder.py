import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet50_Weights

class ResNet50(nn.Module):
    def __init__(self, num_classes, in_features):
        super().__init__()
        
        self.resnet = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        
        # Set channels
        self.resnet.conv1 = torch.nn.Conv2d(in_channels=1,  # same os color channels
                                            out_channels=64,
                                            kernel_size=(5, 5),
                                            stride=2,
                                            padding=0,
                                            bias=True)
        
        self.fc = nn.Linear(in_features=in_features,
                            out_features=num_classes)

    def forward(self, x):
        self.resnet.eval()
        x = self.resnet(x.unsqueeze(0))
        x = self.fc(x)
        return x