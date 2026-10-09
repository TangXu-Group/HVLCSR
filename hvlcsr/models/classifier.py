   

import torch
import torch.nn as nn

class ClassifierHeads(nn.Module):
                                                                     

    def __init__(self, num_classes: int, d: int = 512):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(num_classes, d) * 0.02)
        self.bias = nn.Parameter(torch.zeros(num_classes))

    def forward(self, enhanced: torch.Tensor) -> torch.Tensor:
                                  
        return (enhanced * self.weight.unsqueeze(0)).sum(dim=-1) + self.bias
