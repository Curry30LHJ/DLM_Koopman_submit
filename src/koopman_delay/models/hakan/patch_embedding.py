"""Patch extraction, embedding, and position embedding."""
from __future__ import annotations
import torch
from torch import nn
class PatchEmbedding(nn.Module):
    def __init__(self,channels,context_length,patch_len=5,stride=2,d_model=32):
        super().__init__(); self.patch_len,self.stride=patch_len,stride; self.patch_num=(context_length-patch_len)//stride+1; self.projection=nn.Linear(patch_len,d_model); self.position=nn.Parameter(torch.randn(1,self.patch_num,d_model)*.02)
    def forward(self,x):
        patches=x.unfold(-1,self.patch_len,self.stride); embedded=self.projection(patches); b,c,p,d=embedded.shape; return (embedded.reshape(b*c,p,d)+self.position).reshape(b,c,p,d)
