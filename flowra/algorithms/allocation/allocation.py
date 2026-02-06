"""
Rank Allocation Algorithms

Spectral-Aware Rank Allocation
"""

import torch
from typing import Dict, Tuple
from abc import ABC, abstractmethod
import numpy as np


class RankAllocator(ABC):
    """Base class for rank allocation."""
    
    @abstractmethod
    def allocate(self, conductances: Dict[str, float], layer_shapes: Dict[str, Tuple], budget: int) -> Dict[str, int]:
        pass


class SpectralRankAllocator(RankAllocator):
    """
    Spectral-Aware Rank Allocation
    
    Allocates ranks based on flow conductance γ using water-filling.
    """
    
    def __init__(self, min_rank: int = 1, max_rank_ratio: float = 0.5):
        self.min_rank = min_rank
        self.max_rank_ratio = max_rank_ratio
    
    def allocate(self, conductances: Dict[str, float], layer_shapes: Dict[str, Tuple], budget: int) -> Dict[str, int]:
        if not conductances:
            return {}
        
        # Normalize conductances
        total = sum(conductances.values())
        if total <= 0:
            total = len(conductances)
            normalized = {n: 1.0/total for n in conductances}
        else:
            normalized = {n: g/total for n, g in conductances.items()}
        
        # Compute layer sizes
        sizes = {}
        total_size = 0
        for name, shape in layer_shapes.items():
            m = shape[0]
            n = int(np.prod(shape[1:])) if len(shape) > 1 else 1
            sizes[name] = m * n
            total_size += m * n
        
        # Initial allocation
        allocation = {}
        for name in conductances:
            gamma_norm = normalized[name]
            size_norm = sizes[name] / total_size if total_size > 0 else 1.0
            
            raw = budget * gamma_norm * size_norm
            
            shape = layer_shapes[name]
            m = shape[0]
            n = int(np.prod(shape[1:])) if len(shape) > 1 else 1
            max_rank = int(min(m, n) * self.max_rank_ratio)
            
            allocation[name] = max(self.min_rank, min(max_rank, int(raw)))
        
        # Water-filling adjustment
        allocation = self._water_fill(allocation, layer_shapes, budget)
        
        return allocation
    
    def _water_fill(self, allocation: Dict[str, int], shapes: Dict[str, Tuple], budget: int) -> Dict[str, int]:
        """Adjust to meet budget."""
        def count_params(allocs):
            total = 0
            for name, rank in allocs.items():
                shape = shapes[name]
                m = shape[0]
                n = int(np.prod(shape[1:])) if len(shape) > 1 else 1
                total += rank * (m + n)
            return total
        
        current = count_params(allocation)
        
        for _ in range(1000):
            diff = current - budget
            if abs(diff) < 100:
                break
            
            if diff > 0:
                candidates = [(n, r) for n, r in allocation.items() if r > self.min_rank]
                if not candidates:
                    break
                name, rank = min(candidates, key=lambda x: x[1])
                allocation[name] = rank - 1
            else:
                candidates = []
                for name, rank in allocation.items():
                    shape = shapes[name]
                    m = shape[0]
                    n = int(np.prod(shape[1:])) if len(shape) > 1 else 1
                    max_r = int(min(m, n) * self.max_rank_ratio)
                    if rank < max_r:
                        candidates.append((name, rank))
                if not candidates:
                    break
                name, rank = max(candidates, key=lambda x: x[1])
                allocation[name] = rank + 1
            
            current = count_params(allocation)
        
        return allocation


class UniformRankAllocator(RankAllocator):
    """Allocate uniform rank to all layers."""
    
    def __init__(self, rank: int = 8):
        self.rank = rank
    
    def allocate(self, conductances: Dict[str, float], layer_shapes: Dict[str, Tuple], budget: int) -> Dict[str, int]:
        return {name: self.rank for name in conductances}


class AdaptiveRankAllocator(SpectralRankAllocator):
    """Adaptive allocation that can adjust during training."""
    
    def __init__(self, min_rank: int = 1, max_rank_ratio: float = 0.5, adaptation_rate: float = 0.1):
        super().__init__(min_rank, max_rank_ratio)
        self.adaptation_rate = adaptation_rate
        self.history: Dict[str, list] = {}
    
    def update(self, name: str, importance: float):
        """Update layer importance estimate."""
        if name not in self.history:
            self.history[name] = []
        self.history[name].append(importance)
