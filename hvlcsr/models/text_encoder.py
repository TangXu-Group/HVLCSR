   

from typing import List

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

BERT_NAME = "bert-base-uncased"
BERT_DIM = 768

class TextEncoder(nn.Module):
                                                            

    def __init__(self, label_names: List[str], d: int = 512,
                 model_name: str = BERT_NAME):
        super().__init__()
        self.d = d
        self.label_names = list(label_names)

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        bert = AutoModel.from_pretrained(model_name)
        bert.eval()
        for p in bert.parameters():
            p.requires_grad_(False)

        with torch.no_grad():
            base = self._embed_labels(tokenizer, bert, self.label_names)            
                                                                     
        self.register_buffer("base_embeddings", base)

                                                                           
                                               
        self.proj = nn.Linear(BERT_DIM, d)

    @staticmethod
    def _embed_labels(tokenizer, bert, label_names: List[str]) -> torch.Tensor:
                                                                        
        enc = tokenizer(
            label_names, padding=True, return_tensors="pt",
            add_special_tokens=True,
        )
        out = bert(**enc).last_hidden_state               
        mask = enc["attention_mask"].unsqueeze(-1).to(out.dtype)             
        return (out * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)

    @property
    def num_classes(self) -> int:
        return len(self.label_names)

    def forward(self) -> torch.Tensor:
                                                         
        return self.proj(self.base_embeddings)
