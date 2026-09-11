"""Reversible instance normalization matching the official HaKAN data flow."""
from __future__ import annotations
import torch
from torch import nn
class RevIN(nn.Module):
    def __init__(self, num_features, eps=1e-5, affine=True, subtract_last=False):
        super().__init__(); self.eps,self.affine,self.subtract_last=eps,affine,subtract_last
        if affine: self.affine_weight=nn.Parameter(torch.ones(num_features)); self.affine_bias=nn.Parameter(torch.zeros(num_features))
    def norm(self,x):
        self.last=x[:,-1:].detach() if self.subtract_last else None; self.mean=None if self.subtract_last else x.mean(dim=1,keepdim=True).detach(); self.stdev=(x.var(dim=1,keepdim=True,unbiased=False)+self.eps).sqrt().detach(); y=(x-(self.last if self.subtract_last else self.mean))/self.stdev
        return y*self.affine_weight+self.affine_bias if self.affine else y
    def denorm(self,x):
        y=(x-self.affine_bias)/(self.affine_weight+self.eps**2) if self.affine else x
        return y*self.stdev+(self.last if self.subtract_last else self.mean)
