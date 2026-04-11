# model.py
# CNN architecture — identical to training notebook

import torch
import torch.nn as nn


class ASDClassifierCNN(nn.Module):
    """
    5-layer CNN for ASD detection from sMRI 2D axial slices.
    Input : (B, 3, 224, 224)
    Output: (B, 2)  raw logits
    """
    def __init__(self):
        super().__init__()
        # Block 1
        self.conv1  = nn.Conv2d(3,   16,  3, padding=1)
        self.lrelu1 = nn.LeakyReLU(0.1)
        self.pool1  = nn.MaxPool2d(2, 2)
        self.drop1  = nn.Dropout2d(0.2)
        # Block 2
        self.conv2  = nn.Conv2d(16,  32,  3, padding=1)
        self.lrelu2 = nn.LeakyReLU(0.1)
        self.bn1    = nn.BatchNorm2d(32)
        self.pool2  = nn.MaxPool2d(2, 2)
        self.drop2  = nn.Dropout2d(0.2)
        # Block 3
        self.conv3  = nn.Conv2d(32,  64,  3, padding=1)
        self.lrelu3 = nn.LeakyReLU(0.1)
        self.bn2    = nn.BatchNorm2d(64)
        self.pool3  = nn.MaxPool2d(2, 2)
        self.drop3  = nn.Dropout2d(0.2)
        # Block 4
        self.conv4  = nn.Conv2d(64,  128, 3, padding=1)
        self.lrelu4 = nn.LeakyReLU(0.1)
        self.bn3    = nn.BatchNorm2d(128)
        self.pool4  = nn.MaxPool2d(2, 2)
        self.drop4  = nn.Dropout2d(0.2)
        # Block 5  ← GradCAM target layer
        self.conv5  = nn.Conv2d(128, 256, 3, padding=1)
        self.lrelu5 = nn.LeakyReLU(0.1)
        self.bn4    = nn.BatchNorm2d(256)
        self.pool5  = nn.MaxPool2d(2, 2)
        self.drop5  = nn.Dropout2d(0.2)
        # Classifier head
        self.flatten  = nn.Flatten()
        self.fc1      = nn.Linear(7 * 7 * 256, 100)
        self.lrelu_fc = nn.LeakyReLU(0.1)
        self.fc2      = nn.Linear(100, 2)

    def forward(self, x):
        x = self.drop1(self.pool1(self.lrelu1(self.conv1(x))))
        x = self.drop2(self.pool2(self.bn1(self.lrelu2(self.conv2(x)))))
        x = self.drop3(self.pool3(self.bn2(self.lrelu3(self.conv3(x)))))
        x = self.drop4(self.pool4(self.bn3(self.lrelu4(self.conv4(x)))))
        x = self.drop5(self.pool5(self.bn4(self.lrelu5(self.conv5(x)))))
        x = self.flatten(x)
        x = self.lrelu_fc(self.fc1(x))
        return self.fc2(x)


def load_model(weights_path: str, device: str = 'cpu'):
    """Load trained weights and return model in eval mode."""
    model = ASDClassifierCNN()
    model.load_state_dict(
        torch.load(weights_path, map_location=device, weights_only=True)
    )
    model.eval()
    return model