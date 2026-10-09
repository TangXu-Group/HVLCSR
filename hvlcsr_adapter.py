   

import sys
from pathlib import Path

import torch
import torch.nn as nn

                                                              
_PKG_PARENT = str(Path(__file__).resolve().parent)
if _PKG_PARENT not in sys.path:
    sys.path.insert(0, _PKG_PARENT)

from hvlcsr.labels import LABEL_SETS              
from hvlcsr.losses import hvlcsr_loss              
from hvlcsr.model import HVLCSR              

                                                  
_DATASET_MAP = {"ucm": "ml_ucm", "aid": "ml_aid", "dfc15": "ml_dfc"}

                                                                   
                                               
_ACTIVE_MODEL = None

class HVLCSRAdapter(nn.Module):
                                                           

    def __init__(self, cfg):
        super().__init__()
        global _ACTIVE_MODEL
        dataset = _DATASET_MAP[cfg.data]
                                                             
                                           
                                                                    
                                                             
        use_mpvlm = not getattr(cfg, "no_mpvlm", False)
        use_csrbm = use_mpvlm and not getattr(cfg, "no_csrbm", False)
        mpvlm_leak = getattr(cfg, "mpvlm_leak", 0.0)
        csrbm_detach = getattr(cfg, "csrbm_detach", False)
        csrbm_grad_scale = getattr(cfg, "csrbm_grad_scale", None)
        self.inner = HVLCSR(
            LABEL_SETS[dataset],
            dataset=dataset,
            d=512,
            text_encoder="bert",
            use_csrbm=use_csrbm,
            use_mpvlm=use_mpvlm,
            mpvlm_leak=mpvlm_leak,
            csrbm_detach=csrbm_detach,
            csrbm_grad_scale=csrbm_grad_scale,
            split_zscore=getattr(cfg, "split_zscore", False),
        )
        self._cached_scene_score = None
        self._cached_v = None
        _ACTIVE_MODEL = self

    def forward(self, imgs: torch.Tensor) -> torch.Tensor:
                                                                         
        out = self.inner(imgs, labels=None)
                                                                        
                                              
        self._cached_scene_score = out["scene_score"]
        v = out.get("v")
        self._cached_v = v.detach() if v is not None else None
        return out["logits"]

def model_hvlcsr(cfg) -> HVLCSRAdapter:
    return HVLCSRAdapter(cfg)

def hvlcsr_loss_fn(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
       
    model = _ACTIVE_MODEL
    if (
        model is not None
        and model.training
        and model._cached_v is not None
        and model._cached_scene_score is not None
    ):
                                                                     
                                                                
        model.inner.csrbm.category_analysis.update(
            model._cached_v, targets, model._cached_scene_score
        )
    q = model._cached_scene_score if model is not None else None
    return hvlcsr_loss(logits, targets, scene_score=q)["total"]
