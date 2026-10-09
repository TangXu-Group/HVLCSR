   

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

class MPVLM(nn.Module):
                                                        

    def __init__(self, d: int = 512, eps: float = 1e-8, leak: float = 0.0):
        super().__init__()
        self.d = d
        self.eps = eps
                                                      
                                                               
                                                                          
                                                                        
                                                                          
                                                                        
                                                                     
                                                            
        self.leak = leak

                                                                        
                                                         
                                                                        
    def activation_maps(
        self, feats: List[torch.Tensor], prototypes: torch.Tensor
    ) -> List[torch.Tensor]:
                                                       
        s = F.normalize(prototypes, dim=-1)          
        maps = []
        for feat in feats:
            f = F.normalize(feat, dim=1)                
            a = torch.einsum("cd,bdhw->bchw", s, f)
            maps.append(a)
        return maps

                                                                        
                                                               
                                                                        
    def fuse_scales(
        self, maps: List[torch.Tensor], out_size: Tuple[int, int]
    ) -> torch.Tensor:
                                                                            
        resized = [
            F.interpolate(a, size=out_size, mode="bilinear", align_corners=False)
            if a.shape[-2:] != out_size
            else a
            for a in maps
        ]
        stacked = torch.stack(resized, dim=2)                     

        b, c, l, h, w = stacked.shape
        flat = stacked.reshape(b, c, l, h * w)
        conf = flat.softmax(dim=-1)                                 
        weight = conf.var(dim=-1)                                         
        weight = weight.softmax(dim=2)                                         

        return (stacked * weight.unsqueeze(-1).unsqueeze(-1)).sum(dim=2)

                                                                        
                                                          
                                                                        
    def category_interaction_norm(self, a: torch.Tensor) -> torch.Tensor:
                                           
        c = a.shape[1]
        others = (a.sum(dim=1, keepdim=True) - a) / (c - 1)                            
        r = F.relu(a - others)
        if self.leak > 0:
            r = r + self.leak * F.relu(a)                              
        peak = r.amax(dim=(-2, -1), keepdim=True).clamp(min=self.eps)
        return r / peak

                                                                        
                                                           
                                                                        
    def class_activation_vectors(
        self, a_norm: torch.Tensor, f4: torch.Tensor
    ) -> torch.Tensor:
                                                              
        modulated = a_norm.unsqueeze(2) * f4.unsqueeze(1)                     
        return modulated.mean(dim=(-2, -1))                        

                                                                        
    def forward(
        self, feats: List[torch.Tensor], prototypes: torch.Tensor
    ) -> torch.Tensor:
        f4 = feats[-1]
        maps = self.activation_maps(feats, prototypes)
        fused = self.fuse_scales(maps, out_size=f4.shape[-2:])
        a_norm = self.category_interaction_norm(fused)
        return self.class_activation_vectors(a_norm, f4)
