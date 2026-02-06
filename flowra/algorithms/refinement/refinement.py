"""
Subspace Refinement Algorithms

Progressive Subspace Refinement
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional
from abc import ABC, abstractmethod
import math


class SubspaceRefiner(ABC):
    """Base class for subspace refinement."""
    
    @abstractmethod
    def refine(self, adapters: Dict[str, nn.Module], step: int) -> Dict[str, int]:
        pass


class ProgressiveSubspaceRefiner(SubspaceRefiner):
    """
    Progressive Subspace Refinement
    
    Tracks effective gradient dimension and prunes inactive components.
    """
    
    def __init__(self, epsilon: float = 0.1, warmup: int = 100, min_rank_fraction: float = 0.25):
        self.epsilon = epsilon
        self.warmup = warmup
        self.min_rank_fraction = min_rank_fraction
        self.effective_dims: List[float] = []
        self.prev_dim: Optional[float] = None
        self.initial_ranks: Dict[str, int] = {}
    
    def compute_effective_dim(self, adapters: Dict[str, nn.Module]) -> float:
        """Compute global effective gradient dimension."""
        all_sv = []
        for adapter in adapters.values():
            if hasattr(adapter, 'B') and hasattr(adapter, 'A'):
                with torch.no_grad():
                    W = adapter.B @ adapter.A
                    try:
                        _, sv, _ = torch.linalg.svd(W.float(), full_matrices=False)
                        all_sv.append(sv.cpu())
                    except:
                        pass
        
        if not all_sv:
            return 1.0
        
        sv = torch.cat(all_sv).clamp(min=1e-10)
        return ((sv.sum() ** 2) / (sv ** 2).sum()).item()
    
    def refine(self, adapters: Dict[str, nn.Module], step: int) -> Dict[str, int]:
        if step < self.warmup:
            return {}
        
        # Store initial ranks
        if not self.initial_ranks:
            for name, adapter in adapters.items():
                if hasattr(adapter, 'rank'):
                    self.initial_ranks[name] = adapter.rank
        
        d_eff = self.compute_effective_dim(adapters)
        self.effective_dims.append(d_eff)
        
        if self.prev_dim is not None:
            change = abs(d_eff - self.prev_dim)
            if change < self.epsilon * self.prev_dim:
                return {}
        
        self.prev_dim = d_eff
        changes = {}
        
        for name, adapter in adapters.items():
            if not hasattr(adapter, 'B') or not hasattr(adapter, 'A'):
                continue
            
            with torch.no_grad():
                B, A = adapter.B.data, adapter.A.data
                try:
                    W = B @ A
                    U, sv, Vh = torch.linalg.svd(W.float(), full_matrices=False)
                except:
                    continue
                
                if sv[0] > 0:
                    relative = sv / sv[0]
                    threshold = d_eff / self.initial_ranks.get(name, adapter.rank)
                    keep = (relative >= threshold).sum().item()
                    
                    min_rank = max(1, int(self.initial_ranks.get(name, adapter.rank) * self.min_rank_fraction))
                    new_rank = max(min_rank, keep)
                    
                    if new_rank < adapter.rank:
                        adapter.B.data = (U[:, :new_rank] @ torch.diag(sv[:new_rank].sqrt())).to(adapter.B.dtype)
                        adapter.A.data = (torch.diag(sv[:new_rank].sqrt()) @ Vh[:new_rank, :]).to(adapter.A.dtype)
                        adapter.rank = new_rank
                        changes[name] = new_rank
        
        return changes


class GradientSubspaceTracker:
    """Track gradient subspace evolution."""
    
    def __init__(self, max_rank: int = 100, decay: float = 0.99):
        self.max_rank = max_rank
        self.decay = decay
        self.U: Optional[torch.Tensor] = None
        self.S: Optional[torch.Tensor] = None
        self.step = 0
    
    def update(self, gradient: torch.Tensor):
        grad = gradient.flatten().float()
        
        if self.U is None:
            self.U = grad.unsqueeze(1) / (grad.norm() + 1e-10)
            self.S = torch.tensor([grad.norm()])
        else:
            proj = self.U.T @ grad
            residual = grad - self.U @ proj
            res_norm = residual.norm()
            
            if res_norm > 0.01 * grad.norm() and self.U.size(1) < self.max_rank:
                self.U = torch.cat([self.U, residual.unsqueeze(1) / res_norm], dim=1)
                self.S = torch.cat([self.decay * self.S, res_norm.unsqueeze(0)])
            else:
                self.S = self.decay * self.S
        
        self.step += 1
    
    def get_effective_rank(self) -> float:
        if self.S is None:
            return 0.0
        s_sum = self.S.sum()
        s_sq = (self.S ** 2).sum()
        return (s_sum ** 2 / s_sq).item() if s_sq > 0 else 0.0
