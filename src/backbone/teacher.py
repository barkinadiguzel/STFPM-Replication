import torch
import torch.nn as nn
import torchvision.models as models

class Teacher(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        backbone = models.resnet18(pretrained=pretrained)
        self.features = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,  
            backbone.layer2,  
            backbone.layer3,  
        )
        self.eval()  
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, x):
        feats = []
        for i, layer in enumerate(self.features):
            x = layer(x)
            if i >= 4:  
                feats.append(x)
        return feats   
