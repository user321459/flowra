"""
Redundancy Analysis

Analyzes how amenable layers are to low-rank approximation
by examining their singular value distributions.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple
import math


def compute_effective_rank(matrix: torch.Tensor, threshold: float = 0.01) -> float:
    """
    Compute effective rank of a matrix.
    
    effective_rank = (Σ σ_i)² / (Σ σ_i²)
    
    This equals the full rank when all singular values are equal,
    and approaches 1 when one singular value dominates.
    """
    if matrix.dim() > 2:
        matrix = matrix.view(matrix.size(0), -1)
    
    try:
        _, sv, _ = torch.linalg.svd(matrix.float(), full_matrices=False)
        sv = sv[sv > threshold]
        
        if len(sv) == 0:
            return 0.0
        
        sv_sum = sv.sum()
        sv_sq_sum = (sv ** 2).sum()
        
        if sv_sq_sum > 0:
            return (sv_sum ** 2 / sv_sq_sum).item()
        return 0.0
    except Exception:
        return 0.0


def compute_redundancy_coefficient(matrix: torch.Tensor, threshold: float = 0.01) -> float:
    """
    Compute redundancy coefficient ρ.
    
    ρ = 1 - (effective_rank / full_rank)
    
    Higher ρ indicates more redundancy (better for low-rank adaptation).
    """
    if matrix.dim() > 2:
        matrix = matrix.view(matrix.size(0), -1)
    
    full_rank = min(matrix.shape)
    effective = compute_effective_rank(matrix, threshold)
    
    if full_rank > 0:
        return max(0.0, 1.0 - effective / full_rank)
    return 0.0


class RedundancyAnalyzer:
    """
    Analyzes redundancy patterns across model layers.
    
    Identifies which layers can be effectively compressed
    via low-rank approximation.
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        energy_threshold: float = 0.95
    ):
        self.model = model
        self.device = device
        self.energy_threshold = energy_threshold
        
        self._redundancy_cache: Dict[str, float] = {}
        self._effective_ranks: Dict[str, float] = {}
        self._singular_values: Dict[str, torch.Tensor] = {}
    
    def analyze(self) -> Dict[str, float]:
        """Analyze redundancy for all adaptable layers."""
        self.model.eval()
        
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
                weight = module.weight.detach()
                
                if weight.dim() > 2:
                    weight = weight.view(weight.size(0), -1)
                
                # Compute SVD
                try:
                    _, sv, _ = torch.linalg.svd(weight.float(), full_matrices=False)
                    self._singular_values[name] = sv.cpu()
                    
                    # Effective rank
                    sv_sum = sv.sum()
                    sv_sq_sum = (sv ** 2).sum()
                    if sv_sq_sum > 0:
                        self._effective_ranks[name] = (sv_sum ** 2 / sv_sq_sum).item()
                    else:
                        self._effective_ranks[name] = 1.0
                    
                    # Redundancy coefficient
                    full_rank = min(weight.shape)
                    self._redundancy_cache[name] = max(
                        0.0, 
                        1.0 - self._effective_ranks[name] / full_rank
                    )
                except Exception:
                    self._effective_ranks[name] = weight.size(0)
                    self._redundancy_cache[name] = 0.0
        
        return self._redundancy_cache
    
    def get_redundancy(self, layer_name: str) -> float:
        """Get redundancy coefficient for a layer."""
        return self._redundancy_cache.get(layer_name, 0.0)
    
    def get_effective_rank(self, layer_name: str) -> float:
        """Get effective rank for a layer."""
        return self._effective_ranks.get(layer_name, 0.0)
    
    def get_optimal_rank(
        self, 
        layer_name: str, 
        energy_threshold: float = None
    ) -> int:
        """
        Get optimal rank that captures specified energy.
        
        Returns the smallest rank r such that
        Σ_{i=1}^r σ_i² / Σ σ_i² >= energy_threshold
        """
        if energy_threshold is None:
            energy_threshold = self.energy_threshold
        
        if layer_name not in self._singular_values:
            return 1
        
        sv = self._singular_values[layer_name]
        total_energy = (sv ** 2).sum()
        
        if total_energy == 0:
            return 1
        
        cumulative_energy = (sv ** 2).cumsum(0) / total_energy
        
        # Find first index where cumulative energy exceeds threshold
        indices = (cumulative_energy >= energy_threshold).nonzero()
        
        if len(indices) > 0:
            return int(indices[0].item()) + 1
        return len(sv)
    
    def get_layer_summary(self, layer_name: str) -> Dict:
        """Get comprehensive summary for a layer."""
        return {
            "redundancy": self._redundancy_cache.get(layer_name, 0.0),
            "effective_rank": self._effective_ranks.get(layer_name, 0.0),
            "optimal_rank_95": self.get_optimal_rank(layer_name, 0.95),
            "optimal_rank_99": self.get_optimal_rank(layer_name, 0.99),
        }
