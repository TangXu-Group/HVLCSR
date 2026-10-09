   

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

                                                                        
                             
                                                                        
class SceneDecision(nn.Module):
                                                                    

    def __init__(self, d: int, K: int):
        super().__init__()
        self.K = K
        self.discriminator = nn.Linear(d, K)
                                                                     
                                          
        self.scene_prototypes = nn.Parameter(torch.randn(K, d) * 0.02)

    def forward(self, global_feat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
                                                           
        q = self.discriminator(global_feat).softmax(dim=-1)
        g_bar = self.scene_prototypes.unsqueeze(0) + global_feat.unsqueeze(1)
        return q, g_bar

                                                                        
                                
                                                                        
class CategoryAnalysis(nn.Module):
       

    def __init__(self, c: int, d: int, K: int, alpha: float = 0.5,
                 split_zscore: bool = False):
        super().__init__()
        self.c, self.d, self.K, self.alpha = c, d, K, alpha
        self.n_typical = max(1, int(alpha * c))
                                                                  
                                                                        
                                                                     
                                                                        
                                                                           
        self.split_zscore = split_zscore

        self.register_buffer("cls_feat_sum", torch.zeros(c, d))
        self.register_buffer("cls_count", torch.zeros(c))
        self.register_buffer("cls_intra_cos_sum", torch.zeros(c))
        self.register_buffer("scene_cooc", torch.zeros(K, c))                

    @torch.no_grad()
    def update(self, v: torch.Tensor, labels: torch.Tensor, q: torch.Tensor):
           
        v = v.detach()
        mask = labels > 0.5          
        count_new = mask.sum(dim=0)        

                                                                      
                                                
        proto = self.cls_feat_sum / self.cls_count.clamp(min=1.0).unsqueeze(-1)
        seen = self.cls_count > 0
        if seen.any():
            cos = F.cosine_similarity(v, proto.unsqueeze(0), dim=-1)          
            cos = torch.nan_to_num(cos, nan=0.0)
            intra_new = (cos * mask * seen.unsqueeze(0)).sum(dim=0)
            self.cls_intra_cos_sum += intra_new

                                                          
        self.cls_feat_sum += (v * mask.unsqueeze(-1)).sum(dim=0)
        self.cls_count += count_new

                                                                      
                                         
        k_star = q.detach().argmax(dim=-1)        
        self.scene_cooc.index_add_(
            0, k_star, labels.detach().to(self.scene_cooc.dtype)
        )

    def discrimination_scores(self) -> torch.Tensor:
                                                                        
        count = self.cls_count.clamp(min=1.0)
        proto = self.cls_feat_sum / count.unsqueeze(-1)          
        valid = self.cls_count > 0                                

                                                                       
        pn = F.normalize(proto, dim=-1)
        sim = pn @ pn.T          
        sim = torch.nan_to_num(sim, nan=0.0)
        n_other = (valid.sum() - 1).clamp(min=1)
        o_inter = (sim * valid.unsqueeze(0)).sum(dim=1) / n_other        

                                                        
        o_intra = self.cls_intra_cos_sum / count        

                                                                       
        row_sum = self.scene_cooc.sum(dim=-1, keepdim=True)
        p = torch.where(
            row_sum > 0,
            self.scene_cooc / row_sum.clamp(min=1e-8),
            torch.full_like(self.scene_cooc, 1.0 / self.c),
        )

        d_score = o_inter - o_intra        
        d_score = torch.where(valid, d_score, torch.full_like(d_score, -1e4))
        if self.split_zscore:
                                                                    
            dv = d_score[valid]
            if dv.numel() > 1:
                d_score = torch.where(
                    valid,
                    (d_score - dv.mean()) / dv.std().clamp(min=1e-8),
                    d_score,
                )
            p_mean = p.mean(dim=-1, keepdim=True)
            p_std = p.std(dim=-1, keepdim=True).clamp(min=1e-8)
            p = (p - p_mean) / p_std
        return d_score.unsqueeze(0) + p          

    def typical_masks(self) -> torch.Tensor:
                                                                     
        scores = self.discrimination_scores()
        top = scores.topk(self.n_typical, dim=-1).indices                  
        mask = torch.zeros_like(scores, dtype=torch.bool)
        mask.scatter_(1, top, True)
        return mask

    def forward(self, v, labels, q):
        if self.training and labels is not None:
            self.update(v, labels, q)
        return self.typical_masks()

                                                                        
                                                         
                                                                        
class TreeSemanticBlock(nn.Module):
       

    def __init__(
        self,
        c: int,
        K: int,
        d: int = 512,
        d_state: int = 16,
        d_conv: int = 3,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dt_init_floor: float = 1e-4,
    ):
        super().__init__()
        self.c, self.K, self.d, self.N = c, K, d, d_state
        self.dt_rank = math.ceil(d / 16)

                                                             
        self.in_proj = nn.Linear(d, 2 * d, bias=False)
        self.conv1d = nn.Conv1d(
            d, d, kernel_size=d_conv, padding=(d_conv - 1) // 2,
            groups=d, bias=True,
        )
        self.act = nn.SiLU()

                                                      
        self.x_proj = nn.Linear(d, self.dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, d, bias=True)
        self._init_dt_proj(dt_min, dt_max, dt_init_floor)

                                                                      
                                                            
        a = torch.arange(1, d_state + 1, dtype=torch.float32).log()
        self.A_logs = nn.Parameter(a.view(1, 1, 1, -1).expand(K, c, d, -1).contiguous())

                                    
        self.Ds = nn.Parameter(torch.ones(d))

                                                               
                                                                       
                                                                    
                                                                       
                                                      
        self.typical_gate = nn.Parameter(torch.ones(K))

        self.norm = nn.LayerNorm(d)                                    
        self.out_proj = nn.Linear(d, d, bias=False)
                                                                      
                                                               
                                                           
        nn.init.zeros_(self.out_proj.weight)

    def _init_dt_proj(self, dt_min, dt_max, dt_init_floor):
                                                                        
        dt_init_std = self.dt_rank ** -0.5
        nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        dt = torch.exp(
            torch.rand(self.d) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)

                                                                        
    def generate_params(self, seq: torch.Tensor):
                                                                          
        xz = self.in_proj(seq)                                   
        x, z = xz.chunk(2, dim=-1)                                   
        x = self.conv1d(x.transpose(1, 2)).transpose(1, 2)
        x = self.act(x)
        z = self.act(z)

        xbc = self.x_proj(x)                                               
        dt_raw, b_mat, c_mat = torch.split(
            xbc, [self.dt_rank, self.N, self.N], dim=-1
        )
        delta = F.softplus(self.dt_proj(dt_raw))                
        return x, delta, b_mat, c_mat, z

                                                                        
    def tree_scan(
        self,
        x_cls: torch.Tensor,                                          
        dt_cls: torch.Tensor,                
        b_cls: torch.Tensor,                 
        c_cls: torch.Tensor,                 
        x_root: torch.Tensor,                                           
        dt_root: torch.Tensor,               
        b_root: torch.Tensor,                
        typical_mask: torch.Tensor,               
        q: torch.Tensor,                               
    ) -> torch.Tensor:
           
        Bsz = x_cls.shape[0]
        A = -torch.exp(self.A_logs.float())                        

                                                     
        h_self = dt_cls.unsqueeze(-1) * b_cls.unsqueeze(2) * x_cls.unsqueeze(-1)
                                                                            

        enhanced = x_cls.new_zeros(Bsz, self.c, self.d)
        for k in range(self.K):
                                                                  
            h_root = (
                dt_root[:, k].unsqueeze(-1)
                * b_root[:, k].unsqueeze(1)
                * x_root[:, k].unsqueeze(-1)
            )
                                                                        
            a_bar = torch.exp(dt_cls.unsqueeze(-1) * A[k].unsqueeze(0))             

                                                                         
            h = a_bar * h_root.unsqueeze(1) + h_self                

                                                              
                                                                            
            tmask = typical_mask[k].to(h.dtype)             
            n_typ = tmask.sum().clamp(min=1.0)
            s_typ = (h_self * a_bar * tmask.view(1, -1, 1, 1)).sum(dim=1) / n_typ
            s_typ = s_typ * self.typical_gate[k]
            h = h + s_typ.unsqueeze(1) * (1.0 - tmask).view(1, -1, 1, 1)

                                                                 
            ev = (h * c_cls.unsqueeze(2)).sum(dim=-1) + self.Ds * x_cls
            enhanced = enhanced + q[:, k].view(-1, 1, 1) * ev

        return enhanced

                                                                        
    def forward(
        self,
        v: torch.Tensor,                       
        g_bar: torch.Tensor,                   
        typical_mask: torch.Tensor,               
        q: torch.Tensor,                    
    ) -> torch.Tensor:
                                                                    
        seq = torch.cat([v, g_bar], dim=1)                        
        x, delta, b_mat, c_mat, z = self.generate_params(seq)

        x_cls, x_root = x[:, : self.c], x[:, self.c :]
        dt_cls, dt_root = delta[:, : self.c], delta[:, self.c :]
        b_cls, b_root = b_mat[:, : self.c], b_mat[:, self.c :]
        c_cls = c_mat[:, : self.c]
        z_cls = z[:, : self.c]

        enhanced = self.tree_scan(
            x_cls, dt_cls, b_cls, c_cls, x_root, dt_root, b_root,
            typical_mask, q,
        )
        enhanced = self.norm(enhanced)                                         
        return self.out_proj(enhanced * z_cls)                                

                                                                        
                                                                
                                                                        
class CSRBM(nn.Module):
                                                                

    def __init__(self, c: int, K: int, d: int = 512, alpha: float = 0.5,
                 d_state: int = 16, split_zscore: bool = False):
        super().__init__()
        self.c, self.K, self.d = c, K, d
        self.scene_decision = SceneDecision(d, K)
        self.category_analysis = CategoryAnalysis(c, d, K, alpha,
                                                  split_zscore=split_zscore)
        self.semantic_modeling = TreeSemanticBlock(c, K, d, d_state=d_state)

    def forward(
        self,
        v: torch.Tensor,
        global_feat: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        q, g_bar = self.scene_decision(global_feat)
        typical_mask = self.category_analysis(v, labels, q)
        enhanced = self.semantic_modeling(v, g_bar, typical_mask, q)
        return enhanced, q, {"typical_mask": typical_mask}
