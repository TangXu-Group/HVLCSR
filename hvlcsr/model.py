   

from typing import Dict, List, Optional

import torch
import torch.nn as nn

from .labels import ALPHA_TYPICAL, NUM_SCENES
from .models.classifier import ClassifierHeads
from .models.csrbm import CSRBM
from .models.mpvlm import MPVLM
from .models.text_encoder import TextEncoder
from .models.visual_encoder import VisualEncoder

class _ModulePlaceholder(nn.Module):
       

    def __init__(self, name: str, file_hint: str):
        super().__init__()
        self.name = name
        self.file_hint = file_hint

    def forward(self, *args, **kwargs):              
        raise NotImplementedError(
            f"{self.name} is not implemented yet. "
            f"Add it in {self.file_hint} and register it in HVLCSR."
        )

class HVLCSR(nn.Module):
                                                                      

    def __init__(
        self,
        label_names: List[str],
        dataset: str = "ml_ucm",
        d: int = 512,
        num_scenes: Optional[int] = None,
        alpha: float = ALPHA_TYPICAL,
        pretrained_backbone: bool = True,
        text_encoder: str = "bert",
        bert_name: str = "bert-base-uncased",
        clip_model: str = "RN50",
        clip_weights_path: Optional[str] = None,
        prompt_template: Optional[str] = None,
        use_csrbm: bool = True,
        use_mpvlm: bool = True,
        mpvlm_leak: float = 0.0,
        csrbm_detach: bool = False,
        csrbm_grad_scale: Optional[float] = None,
        split_zscore: bool = False,
    ):
           
        super().__init__()
        self.label_names = list(label_names)
        self.num_classes = len(label_names)
        self.d = d
        self.K = num_scenes if num_scenes is not None else NUM_SCENES[dataset]
        self.alpha = alpha
        self.use_csrbm = use_csrbm
        self.use_mpvlm = use_mpvlm
        self.csrbm_detach = csrbm_detach
                                                                
                                                                  
        self.csrbm_grad_scale = (
            0.0 if csrbm_detach else csrbm_grad_scale
        )

                                                                        
        self.visual_encoder = VisualEncoder(d=d, pretrained=pretrained_backbone)
                                                                        
                                                                           
        if use_mpvlm:
            if text_encoder == "clip":
                                                                                
                                                                        
                from .models.clip_text_encoder import CLIPTextEncoder
                self.text_encoder = CLIPTextEncoder(
                    label_names, d=d, clip_model=clip_model,
                    weights_path=clip_weights_path, prompt_template=prompt_template,
                )
            elif text_encoder == "bert":
                self.text_encoder = TextEncoder(label_names, d=d, model_name=bert_name)
            else:
                raise ValueError(f"unknown text_encoder: {text_encoder}")

                                                           
        self.mpvlm: nn.Module = MPVLM(d=d, leak=mpvlm_leak)

                                                           
        self.csrbm: nn.Module = CSRBM(c=self.num_classes, K=self.K, d=d,
                                      alpha=self.alpha,
                                      split_zscore=split_zscore)

                                                                         
        self.classifier: nn.Module = ClassifierHeads(
            num_classes=self.num_classes, d=d
        )

                                                                        
                                                                     
                                                                        
    def extract_features(self, images: torch.Tensor):
                                                            
        feats, global_feat = self.visual_encoder(images)
        prototypes = self.text_encoder() if self.use_mpvlm else None
        return feats, global_feat, prototypes

                                                                        
    def forward(self, images: torch.Tensor,
                labels: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
           
        feats, global_feat, prototypes = self.extract_features(images)

        out: Dict[str, torch.Tensor] = {
            "feats": feats,
            "global_feat": global_feat,
            "prototypes": prototypes,
        }

        if not self.use_mpvlm:
                                                                          
                                                                         
                                                                        
            out["enhanced"] = global_feat.unsqueeze(1).expand(
                -1, self.num_classes, -1
            )
            out["scene_score"] = None
            out["logits"] = self.classifier(out["enhanced"])
            return out

                                                                    
        if not isinstance(self.mpvlm, _ModulePlaceholder):
            out["v"] = self.mpvlm(feats, prototypes)
                                                                  
                                                                           
                                                                           
            if self.use_csrbm and not isinstance(self.csrbm, _ModulePlaceholder):
                                                                     
                                                                         
                                                                         
                                                                       
                                                                         
                                                     
                if self.csrbm_grad_scale is None:
                    csrbm_v, csrbm_g = out["v"], global_feat
                else:
                    s = self.csrbm_grad_scale
                    csrbm_v = out["v"] * s + out["v"].detach() * (1 - s)
                    csrbm_g = global_feat * s + global_feat.detach() * (1 - s)
                enhanced, scene_score, info = self.csrbm(
                    csrbm_v, csrbm_g, labels=labels
                )
                                                                       
                                                                      
                                                                         
                                                                         
                out["enhanced"] = enhanced + out["v"]
                out["scene_score"] = scene_score
                out["typical_mask"] = info["typical_mask"]
            else:
                out["enhanced"] = out["v"]
                out["scene_score"] = None
                                                              
            if not isinstance(self.classifier, _ModulePlaceholder):
                out["logits"] = self.classifier(out["enhanced"])

        return out
