"""
Multi-Task Composition Algorithms

Orthogonal Adapter Composition
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
from abc import ABC, abstractmethod
import numpy as np


class AdapterComposer(ABC):
    """Base class for adapter composition."""
    
    @abstractmethod
    def compose(self, adapters: List[Dict[str, nn.Module]], weights: List[float] = None) -> Dict[str, torch.Tensor]:
        pass


class LinearComposer(AdapterComposer):
    """Simple weighted average composition."""
    
    def compose(self, adapters: List[Dict[str, nn.Module]], weights: List[float] = None) -> Dict[str, torch.Tensor]:
        K = len(adapters)
        if weights is None:
            weights = [1.0/K] * K
        
        weights = [w/sum(weights) for w in weights]
        
        all_layers = set()
        for a in adapters:
            all_layers.update(a.keys())
        
        merged = {}
        for layer in all_layers:
            layer_adapts = []
            layer_weights = []
            
            for k, a in enumerate(adapters):
                if layer in a and hasattr(a[layer], 'get_adaptation'):
                    layer_adapts.append(a[layer].get_adaptation().detach())
                    layer_weights.append(weights[k])
            
            if layer_adapts:
                w_norm = [w/sum(layer_weights) for w in layer_weights]
                merged[layer] = sum(w * a for w, a in zip(w_norm, layer_adapts))
        
        return merged


class OrthogonalComposer(AdapterComposer):
    """
    Orthogonal Adapter Composition
    
    Minimizes interference through Gram-Schmidt orthogonalization.
    """
    
    def __init__(self, threshold: float = 0.3, angle_threshold: float = np.pi/4):
        self.threshold = threshold
        self.angle_threshold = angle_threshold
    
    def compute_interference(self, a1: torch.Tensor, a2: torch.Tensor) -> float:
        overlap = a1 * a2
        n1, n2 = a1.norm(), a2.norm()
        if n1 > 0 and n2 > 0:
            return (overlap.norm() / (n1 * n2)).item()
        return 0.0
    
    def orthogonalize(self, a1: torch.Tensor, a2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        v1 = a1.flatten()
        v2 = a2.flatten()
        
        v1_norm = v1 / (v1.norm() + 1e-10)
        proj = torch.dot(v2, v1_norm) * v1_norm
        v2_orth = v2 - proj
        
        return a1, v2_orth.view(a2.shape)
    
    def compose(self, adapters: List[Dict[str, nn.Module]], weights: List[float] = None) -> Dict[str, torch.Tensor]:
        K = len(adapters)
        if K == 0:
            return {}
        
        if weights is None:
            weights = [1.0/K] * K
        weights = [w/sum(weights) for w in weights]
        
        # Find conflicts
        task_vecs = []
        for a in adapters:
            params = []
            for adapter in a.values():
                if hasattr(adapter, 'get_adaptation'):
                    params.append(adapter.get_adaptation().flatten())
            task_vecs.append(torch.cat(params) if params else torch.tensor([0.0]))
        
        conflicts = []
        for i in range(K):
            for j in range(i+1, K):
                interf = self.compute_interference(task_vecs[i], task_vecs[j])
                if interf > self.threshold:
                    conflicts.append((i, j))
        
        # Merge layers
        all_layers = set()
        for a in adapters:
            all_layers.update(a.keys())
        
        merged = {}
        for layer in all_layers:
            layer_adapts = []
            layer_weights = []
            
            for k, a in enumerate(adapters):
                if layer in a and hasattr(a[layer], 'get_adaptation'):
                    layer_adapts.append(a[layer].get_adaptation().detach())
                    layer_weights.append(weights[k])
            
            if not layer_adapts:
                continue
            
            # Orthogonalize conflicts
            for (i, j) in conflicts:
                if i < len(layer_adapts) and j < len(layer_adapts):
                    layer_adapts[i], layer_adapts[j] = self.orthogonalize(layer_adapts[i], layer_adapts[j])
            
            # Weighted merge
            w_norm = [w/sum(layer_weights) for w in layer_weights]
            merged[layer] = sum(w * a for w, a in zip(w_norm, layer_adapts))
        
        return merged


class TaskArithmeticComposer(AdapterComposer):
    """Task arithmetic composition."""
    
    def __init__(self, scaling: float = 1.0):
        self.scaling = scaling
    
    def compose(self, adapters: List[Dict[str, nn.Module]], weights: List[float] = None) -> Dict[str, torch.Tensor]:
        K = len(adapters)
        if weights is None:
            weights = [self.scaling] * K
        
        all_layers = set()
        for a in adapters:
            all_layers.update(a.keys())
        
        merged = {}
        for layer in all_layers:
            combined = None
            for k, a in enumerate(adapters):
                if layer in a and hasattr(a[layer], 'get_adaptation'):
                    adapt = a[layer].get_adaptation().detach()
                    if combined is None:
                        combined = weights[k] * adapt
                    else:
                        combined = combined + weights[k] * adapt
            if combined is not None:
                merged[layer] = combined
        
        return merged
