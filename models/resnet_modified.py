import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet18_Weights


class ResNetBinaryClassifier(nn.Module):
    def __init__(self, pretrained=True):
        super(ResNetBinaryClassifier, self).__init__()

        # Load ResNet18 with pretrained weights
        self.resnet18 = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)

        # Modify the first layer to accept 1-channel input
        self.resnet18.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # Modify the final layer for binary classification
        self.resnet18.fc = nn.Linear(self.resnet18.fc.in_features, 1)

        # Add sigmoid activation for binary output
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.resnet18(x)
        return self.sigmoid(x)