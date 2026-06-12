"""
Jacobian Norm Estimators

Implements methods for estimating layer Jacobian norms:
- Power iteration for spectral norm
- Hutchinson trace estimator
- Analytical bounds for specific layer types

The Jacobian captures how much a layer amplifies input perturbations,
indicating its influence on the overall function.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Optional, Tuple, List, Any
from abc import ABC, abstractmethod
import math


class JacobianEstimator(ABC):
    """
    Abstract base class for Jacobian norm estimators.
    
    Estimates ||J_ℓ||_{2→2} for each layer, which measures the
    spectral norm of the layer's contribution to the network Jacobian.
    """
    
    def __init__(self, model: nn.Module, device: torch.device):
        self.model = model
        self.device = device
        self._jacobian_norms: Dict[str, float] = {}
    
    @abstractmethod
    def estimate(
        self,
        dataloader: DataLoader,
        num_samples: int = 10
    ) -> Dict[str, float]:
        """
        Estimate Jacobian norms for all layers.
        
        Args:
            dataloader: Data for estimation
            num_samples: Number of samples to use
        
        Returns:
            Dict mapping layer names to spectral norms
        """
        pass
    
    def get_jacobian_norm(self, layer_name: str) -> float:
        """Get Jacobian norm for a specific layer."""
        return self._jacobian_norms.get(layer_name, 0.0)
    
    def get_normalized_norms(self) -> Dict[str, float]:
        """Get Jacobian norms normalized by total network norm."""
        if not self._jacobian_norms:
            return {}
        
        total_norm_sq = sum(n ** 2 for n in self._jacobian_norms.values())
        total_norm = math.sqrt(total_norm_sq) if total_norm_sq > 0 else 1.0
        
        return {
            name: norm / total_norm 
            for name, norm in self._jacobian_norms.items()
        }


class PowerIterationEstimator(JacobianEstimator):
    """
    Estimate spectral norm using power iteration.
    
    For a weight matrix W, the spectral norm is:
        ||W||_2 = max ||Wx|| / ||x|| = σ_max(W)
    
    Power iteration converges to this value:
        v_{k+1} = W^T W v_k / ||W^T W v_k||
        ||W||_2 ≈ ||W v_k||
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        num_iterations: int = 10,
        tolerance: float = 1e-4
    ):
        super().__init__(model, device)
        self.num_iterations = num_iterations
        self.tolerance = tolerance
    
    def estimate(
        self,
        dataloader: DataLoader = None,
        num_samples: int = 10
    ) -> Dict[str, float]:
        """
        Estimate spectral norms using power iteration.
        
        Note: This method doesn't actually need data - it operates
        directly on weight matrices.
        """
        self.model.eval()
        
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                norm = self._power_iteration_linear(module.weight)
                self._jacobian_norms[name] = norm
            
            elif isinstance(module, (nn.Conv1d, nn.Conv2d)):
                norm = self._power_iteration_conv(module)
                self._jacobian_norms[name] = norm
            
            elif isinstance(module, (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d)):
                self._jacobian_norms[name] = 1.0
        
        return self._jacobian_norms
    
    def _power_iteration_linear(self, weight: torch.Tensor) -> float:
        """Power iteration for linear layer."""
        with torch.no_grad():
            weight = weight.float()
            m, n = weight.shape
            
            v = torch.randn(n, device=weight.device)
            v = v / v.norm()
            
            prev_norm = 0.0
            
            for _ in range(self.num_iterations):
                u = weight @ v
                v = weight.t() @ u
                
                norm = v.norm()
                v = v / (norm + 1e-10)
                
                if abs(norm - prev_norm) < self.tolerance:
                    break
                prev_norm = norm
            
            u = weight @ v
            spectral_norm = u.norm().item()
            
            return spectral_norm
    
    def _power_iteration_conv(self, module: nn.Module) -> float:
        """Power iteration for convolutional layer."""
        with torch.no_grad():
            weight = module.weight.float()
            weight_2d = weight.view(weight.size(0), -1)
            return self._power_iteration_linear(weight_2d)


class HutchinsonEstimator(JacobianEstimator):
    """
    Estimate Jacobian trace using Hutchinson's trace estimator.
    
    Tr(J^T J) = E[z^T J^T J z] for random z with E[zz^T] = I
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        num_samples: int = 50
    ):
        super().__init__(model, device)
        self.num_samples = num_samples
    
    def estimate(
        self,
        dataloader: DataLoader,
        num_samples: int = None
    ) -> Dict[str, float]:
        """Estimate Jacobian norms using Hutchinson estimator."""
        if num_samples is None:
            num_samples = self.num_samples
        
        self.model.eval()
        
        batch = next(iter(dataloader))
        if isinstance(batch, (tuple, list)):
            sample_input = batch[0][:1].to(self.device)
        else:
            sample_input = batch[:1].to(self.device)
        
        layer_outputs = {}
        hooks = []
        
        def save_output(name):
            def hook(module, input, output):
                layer_outputs[name] = output
            return hook
        
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
                hook = module.register_forward_hook(save_output(name))
                hooks.append(hook)
        
        trace_accum = {}
        
        for _ in range(num_samples):
            layer_outputs.clear()
            
            sample_input.requires_grad_(True)
            output = self.model(sample_input)
            
            z = torch.randn_like(output)
            output.backward(z, retain_graph=True)
            
            for name, layer_out in layer_outputs.items():
                if hasattr(layer_out, 'grad') and layer_out.grad is not None:
                    jvp = layer_out.grad
                    if name not in trace_accum:
                        trace_accum[name] = 0.0
                    trace_accum[name] += (jvp ** 2).sum().item()
            
            self.model.zero_grad()
        
        for hook in hooks:
            hook.remove()
        
        for name in trace_accum:
            frobenius_sq = trace_accum[name] / num_samples
            self._jacobian_norms[name] = math.sqrt(frobenius_sq)
        
        return self._jacobian_norms


class AnalyticalJacobianEstimator(JacobianEstimator):
    """
    Use analytical bounds for Jacobian norms.
    """
    
    def __init__(self, model: nn.Module, device: torch.device):
        super().__init__(model, device)
    
    def estimate(
        self,
        dataloader: DataLoader = None,
        num_samples: int = 10
    ) -> Dict[str, float]:
        """Compute analytical bounds."""
        power_estimator = PowerIterationEstimator(
            self.model, self.device, num_iterations=20
        )
        
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
                self._jacobian_norms[name] = power_estimator._power_iteration_linear(
                    module.weight.view(module.weight.size(0), -1)
                )
            
            elif isinstance(module, (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d)):
                if hasattr(module, 'weight') and module.weight is not None:
                    self._jacobian_norms[name] = module.weight.abs().max().item()
                else:
                    self._jacobian_norms[name] = 1.0
            
            elif isinstance(module, (nn.ReLU, nn.GELU, nn.SiLU)):
                self._jacobian_norms[name] = 1.0
            
            elif isinstance(module, nn.Softmax):
                self._jacobian_norms[name] = 1.0
            
            elif isinstance(module, nn.Dropout):
                p = module.p
                self._jacobian_norms[name] = 1.0 / (1 - p) if p < 1 else 0.0
        
        return self._jacobian_norms
