import torch.nn as nn
import torchvision.models as models

class ResNet50(nn.Module):
    def __init__(self, num_classes, in_features):
        super().__init__()
        self.resnet = models.resnet50(pretrained=True)
        self.fc = nn.Linear(in_features=in_features, num_classes=num_classes)

    def forward(self, x):
        x = self.resnet(x)
        x = self.fc(x)
        return x