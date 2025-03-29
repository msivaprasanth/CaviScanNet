import torch
import torch.nn as nn
import torchvision.models as models

class DentalCariesClassifier(nn.Module):
    """
    Dental Caries Classification model based on ResNet-50.
    Classifies dental X-ray images into three categories:
    - Deep Caries
    - Medium Caries
    - Superficial Caries (includes normal cases)
    """
    def __init__(self, num_classes=4):
        super().__init__()
        # Load pre-trained ResNet-50
        self.model = models.resnet50(pretrained=True)
        # Modify final layers
        in_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.model(x) 