   

from typing import List, Tuple

import torch
import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18

                                 
_STAGE_CHANNELS = (64, 128, 256, 512)

                                                                   
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

class VisualEncoder(nn.Module):
                                                                       

    def __init__(self, d: int = 512, pretrained: bool = True):
        super().__init__()
        self.d = d

        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = resnet18(weights=weights)

                                                                   
        self.stem = nn.Sequential(
            backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool
        )
                                                            
        self.layers = nn.ModuleList(
            [backbone.layer1, backbone.layer2, backbone.layer3, backbone.layer4]
        )
                                                         
        self.projs = nn.ModuleList(
            [nn.Conv2d(c, d, kernel_size=1) for c in _STAGE_CHANNELS]
        )

    def forward(self, images: torch.Tensor) -> Tuple[List[torch.Tensor], torch.Tensor]:
        x = self.stem(images)
        feats: List[torch.Tensor] = []
        for layer, proj in zip(self.layers, self.projs):
            x = layer(x)
            feats.append(proj(x))
        global_feat = feats[-1].mean(dim=(2, 3))            
        return feats, global_feat

def build_image_transform(train: bool = False, image_size: int = 256):
       
    from torchvision import transforms

    ops = [transforms.Resize((image_size, image_size))]
    if train:
        ops.append(transforms.RandomHorizontalFlip())
    ops += [
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]
    return transforms.Compose(ops)
