"""Official-style intra-patch/inter-patch Hahn KAN residual block."""
from __future__ import annotations
from torch import nn
from .hahn_polynomials import HahnPolynomials
class HahnKANBlock(nn.Module):
    def __init__(self,d_model,patch_num): super().__init__(); self.intra_patch_hahn_kan=HahnPolynomials(d_model,d_model); self.inter_patch_hahn_kan=HahnPolynomials(patch_num,patch_num)
    def forward(self,x): return self.inter_patch_hahn_kan(self.intra_patch_hahn_kan(x).transpose(1,2)).transpose(1,2)
