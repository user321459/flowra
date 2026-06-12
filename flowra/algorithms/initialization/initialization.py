"""
Initialization Algorithms

Flow-Aware Parameter Initialization
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Tuple, Optional
from abc import ABC, abstractmethod
import math
import numpy as np


class AdapterInitializer(ABC):
    """Base class for adapter initialization."""
    
    @abstractmethod
    def initialize(self, layer: nn.Module, rank: int) -> Tuple[torch.Tensor, torch.Tensor]:
        pass


class KaimingInitializer(AdapterInitializer):
    """Standard Kaiming initialization."""
    
    def initialize(self, layer: nn.Module, rank: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if isinstance(layer, nn.Linear):
            m, n = layer.out_features, layer.in_features
        else:
            m = layer.weight.size(0)
            n = int(np.prod(layer.weight.shape[1:]))
        
        B = torch.zeros(m, rank)
        A = torch.zeros(rank, n)
        
        std = 1.0 / math.sqrt(n)
        A.uniform_(-std, std)
        
        return B, A


class SVDInitializer(AdapterInitializer):
    """Initialize from SVD of pretrained weights."""
    
    def initialize(self, layer: nn.Module, rank: int) -> Tuple[torch.Tensor, torch.Tensor]:
        weight = layer.weight.detach()
        if weight.dim() > 2:
            weight = weight.view(weight.size(0), -1)
        
        try:
            U, S, Vh = torch.linalg.svd(weight.float(), full_matrices=False)
            
            B = U[:, :rank] * S[:rank].sqrt().unsqueeze(0)
            A = S[:rank].sqrt().unsqueeze(1) * Vh[:rank, :]
            
            # Scale down to start small
            B = B * 0.01
            
            return B.to(weight.dtype), A.to(weight.dtype)
        except:
            return KaimingInitializer().initialize(layer, rank)


class FlowAwareInitializer(AdapterInitializer):
    """
    Flow-Aware Parameter Initialization
    
    Initialize based on gradient covariance.
    """
    
    def __init__(self, model: nn.Module = None, device: torch.device = None):
        self.model = model
        self.device = device or torch.device('cpu')
        self.grad_stats: Dict[str, torch.Tensor] = {}
    
    def collect_gradients(self, dataloader: DataLoader, loss_fn: nn.Module = None, num_samples: int = 50):
        """Collect gradient statistics."""
        if self.model is None:
            return
        
        if loss_fn is None:
            loss_fn = nn.CrossEntropyLoss()
        
        self.model.eval()
        samples = 0
        
        for batch in dataloader:
            if samples >= num_samples:
                break
            
            if isinstance(batch, (tuple, list)):
                inputs = batch[0].to(self.device)
                targets = batch[1].to(self.device) if len(batch) > 1 else None
            else:
                inputs = batch.to(self.device)
                targets = None
            
            self.model.zero_grad()
            outputs = self.model(inputs)
            
            if targets is not None:
                loss = loss_fn(outputs.view(-1, outputs.size(-1)) if outputs.dim() == 3 else outputs,
                              targets.view(-1) if targets.dim() > 1 else targets)
            else:
                loss = outputs.mean()
            
            loss.backward()
            
            for name, module in self.model.named_modules():
                if isinstance(module, nn.Linear) and module.weight.grad is not None:
                    grad_sq = module.weight.grad.detach().view(-1) ** 2
                    if name not in self.grad_stats:
                        self.grad_stats[name] = grad_sq.clone()
                    else:
                        self.grad_stats[name] += grad_sq
            
            samples += inputs.size(0)
        
        for name in self.grad_stats:
            self.grad_stats[name] /= samples
    
    def initialize(self, layer: nn.Module, rank: int, layer_name: str = None) -> Tuple[torch.Tensor, torch.Tensor]:
        if isinstance(layer, nn.Linear):
            m, n = layer.out_features, layer.in_features
        else:
            m = layer.weight.size(0)
            n = int(np.prod(layer.weight.shape[1:]))
        
        if layer_name and layer_name in self.grad_stats:
            grad_cov = self.grad_stats[layer_name]
            
            try:
                grad_2d = grad_cov.view(m, n)
                row_imp = grad_2d.sum(dim=1)
                col_imp = grad_2d.sum(dim=0)
                
                _, top_rows = torch.topk(row_imp, min(rank, m))
                _, top_cols = torch.topk(col_imp, min(rank, n))
                
                B = torch.zeros(m, rank, device=self.device)
                A = torch.zeros(rank, n, device=self.device)
                
                row_w = torch.sqrt(row_imp[top_rows] + 1e-10)
                col_w = torch.sqrt(col_imp[top_cols] + 1e-10)
                
                for i, (idx, w) in enumerate(zip(top_rows, row_w)):
                    if i < rank:
                        B[idx, i] = w
                
                for i, (idx, w) in enumerate(zip(top_cols, col_w)):
                    if i < rank:
                        A[i, idx] = w
                
                # Normalize
                B = B / (B.norm() + 1e-10) * 0.01
                A = A / (A.norm() + 1e-10)
                
                return B.to(layer.weight.dtype), A.to(layer.weight.dtype)
            except:
                pass
        
        return KaimingInitializer().initialize(layer, rank)
    
    def compute_scaling(self, sensitivity: float, rank: int) -> float:
        """Compute flow-aware scaling: α = √(ψ/r)"""
        if sensitivity <= 0 or rank <= 0:
            return 0.01
        return max(0.001, min(1.0, math.sqrt(sensitivity / rank)))
