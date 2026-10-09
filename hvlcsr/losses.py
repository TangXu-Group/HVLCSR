   

from typing import Dict, Optional

import torch
import torch.nn.functional as F

def asymmetric_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    gamma_pos: float = 1.0,
    gamma_neg: float = 4.0,
    clip: float = 0.05,
    eps: float = 1e-8,
) -> torch.Tensor:
                                                           
    p = torch.sigmoid(logits)
    p_pos = p.clamp(min=eps, max=1.0)
                                                                 
                                                       
    p_neg = (p - clip).clamp(min=eps) if clip > 0 else p.clamp(min=eps)

    loss_pos = (1.0 - p) ** gamma_pos * torch.log(p_pos)
    loss_neg = p_neg ** gamma_neg * torch.log((1.0 - p_neg).clamp(min=eps))

    loss = -(targets * loss_pos + (1.0 - targets) * loss_neg)
    return loss.mean()

def scene_decision_loss(q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
                                                                    
    Bsz, K = q.shape
                                                                
    l_sd1 = -(q * (q + eps).log()).sum(dim=-1).mean()
                                                                 
                                                                    
    q_bar = q.mean(dim=0)        
    l_sd2 = (q_bar * (q_bar + eps).log()).sum() + torch.log(
        torch.tensor(float(K), device=q.device)
    )
    return l_sd1 + l_sd2

def hvlcsr_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    scene_score: Optional[torch.Tensor] = None,
    lambda_sd: float = 1.0,
    gamma_pos: float = 1.0,
    gamma_neg: float = 4.0,
    clip: float = 0.05,
) -> Dict[str, torch.Tensor]:
                                                            
    l_asl = asymmetric_loss(logits, targets, gamma_pos, gamma_neg, clip)
    out: Dict[str, torch.Tensor] = {"l_asl": l_asl}
    if scene_score is not None and lambda_sd > 0:
        l_sd = scene_decision_loss(scene_score)
        out["l_sd"] = l_sd
        out["total"] = l_asl + lambda_sd * l_sd
    else:
        out["total"] = l_asl
    return out
