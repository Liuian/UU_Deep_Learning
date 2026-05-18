import torch
import torch.nn as nn
import torch.nn.functional as F

class EEGNet(nn.Module):
    """
    Modern PyTorch implementation of EEGNet for MEG/EEG data.
    Takes input of shape: (Batch, 1, Channels, TimeSteps)
    """
    def __init__(self, num_classes=4, channels=248, samples=3562):
        super(EEGNet, self).__init__()
        
        # Block 1 (Temporal Convolution)
        # Using 64 as the kernel size for the temporal axis
        self.conv1 = nn.Conv2d(1, 16, (1, 64), padding=(0, 32), bias=False)
        self.batchnorm1 = nn.BatchNorm2d(16)
        
        # Block 2 (Spatial Convolution via Depthwise)
        # groups=16 ensures each of the 16 temporal filters gets its own spatial filter
        self.conv2 = nn.Conv2d(16, 32, (channels, 1), bias=False, groups=16)
        self.batchnorm2 = nn.BatchNorm2d(32)
        self.pooling2 = nn.AvgPool2d((1, 4))
        
        # Block 3 (Separable Convolution)
        self.conv3 = nn.Conv2d(32, 32, (1, 16), padding=(0, 8), bias=False, groups=32)
        self.conv4 = nn.Conv2d(32, 32, (1, 1), bias=False)
        self.batchnorm3 = nn.BatchNorm2d(32)
        self.pooling3 = nn.AvgPool2d((1, 8))
        
        # Calculate output size dynamically to avoid hardcoding flattened shape
        with torch.no_grad():
            dummy = torch.zeros(1, 1, channels, samples)
            out = self._forward_features(dummy)
            fc_in = out.view(1, -1).size(1)
            
        # Classification layer
        self.fc1 = nn.Linear(fc_in, num_classes)
        
    def _forward_features(self, x):
        # Block 1
        x = self.conv1(x)
        x = self.batchnorm1(x)
        
        # Block 2
        x = self.conv2(x)
        x = self.batchnorm2(x)
        x = F.elu(x)
        x = F.dropout(x, 0.25)
        x = self.pooling2(x)
        
        # Block 3
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.batchnorm3(x)
        x = F.elu(x)
        x = F.dropout(x, 0.25)
        x = self.pooling3(x)
        
        return x
        
    def forward(self, x):
        x = self._forward_features(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x
