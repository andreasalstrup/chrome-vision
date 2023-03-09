import torch.nn as nn
import torchvision.models as models

class ResNet50(nn.Module):
    def __init__(self, num_classes, in_features):
        super().__init__()
        self.resnet = models.resnet50(pretrained=True)
        self.fc = nn.Linear(in_features=in_features, out_features=num_classes)

    def forward(self, x):
        print(f"ResNet50 input shape 1: {x.shape}")
        x = self.resnet(x)
        print(f"ResNet50 output shape 2: {x.shape}")
        x = self.fc(x)
        return x