"""Finite Hahn-polynomial KAN projection adapted from audited equations."""
from __future__ import annotations
import torch
from torch import nn
class HahnPolynomials(nn.Module):
    def __init__(self,input_dim,output_dim,degree=3,alpha=1.,beta=1.,N=7):
        super().__init__(); self.input_dim,self.degree,self.alpha,self.beta,self.N=input_dim,degree,alpha,beta,N; self.coeffs=nn.Parameter(torch.empty(input_dim,output_dim,degree+1)); nn.init.normal_(self.coeffs,std=1/(input_dim*(degree+1)))
    def forward(self,x):
        shape=x.shape[:-1]; x=torch.tanh(x.reshape(-1,self.input_dim)); p=[torch.ones_like(x)];
        if self.degree>=1: p.append(1-((self.alpha+self.beta+2)*x)/((self.alpha+1)*self.N))
        for n in range(2,self.degree+1):
            m=n-1; A=(m+self.alpha+self.beta+1)*(m+self.alpha+1)*(self.N-m)/((2*m+self.alpha+self.beta+1)*(2*m+self.alpha+self.beta+2)); C=m*(m+self.alpha+self.beta+self.N+1)*(m+self.beta)/((2*m+self.alpha+self.beta)*(2*m+self.alpha+self.beta+1)); p.append(((A+C-x)*p[-1]-C*p[-2])/A)
        return torch.einsum('bid,iod->bo',torch.stack(p,dim=-1),self.coeffs).reshape(*shape,-1)
