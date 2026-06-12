"""
Fisher Information Estimators

Implements various methods for estimating the Fisher Information Matrix:
- Empirical Fisher (gradient outer products)
- Diagonal Fisher (efficient diagonal approximation)
- KFAC (Kronecker-factored approximation)

The Fisher Information Matrix captures how sensitive the loss is to
parameter changes, indicating information transport capacity.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Optional, Tuple, List, Any, Iterator
from abc import ABC, abstractmethod
import math


class FisherEstimator(ABC):
    """
    Abstract base class for Fisher Information estimators.
    
    The Fisher Information Matrix F is defined as:
        F = E[(∇θ log p(y|x))^T (∇θ log p(y|x))]
    
    Different subclasses implement different approximations to this quantity.
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        damping: float = 1e-4
    ):
        self.model = model
        self.device = device
        self.damping = damping
        
        # Cache for Fisher blocks
        self._fisher_cache: Dict[str, torch.Tensor] = {}
        self._sample_count: int = 0
    
    @abstractmethod
    def update(
        self,
        inputs: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        loss: Optional[torch.Tensor] = None
    ) -> None:
        """Update Fisher estimate with a batch of data."""
        pass
    
    @abstractmethod
    def get_fisher_block(self, layer_name: str) -> Optional[torch.Tensor]:
        """Get Fisher block for a specific layer."""
        pass
    
    def compute(
        self,
        dataloader: DataLoader,
        loss_fn: nn.Module,
        num_samples: Optional[int] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute Fisher blocks for all layers.
        
        Args:
            dataloader: Data for Fisher estimation
            loss_fn: Loss function
            num_samples: Maximum samples to use (None = all)
        
        Returns:
            Dict mapping layer names to Fisher blocks
        """
        self.model.eval()
        self._reset()
        
        samples_seen = 0
        
        for batch in dataloader:
            if num_samples and samples_seen >= num_samples:
                break
            
            # Prepare batch
            if isinstance(batch, (tuple, list)):
                inputs = batch[0].to(self.device)
                targets = batch[1].to(self.device) if len(batch) > 1 else None
            else:
                inputs = batch.to(self.device)
                targets = None
            
            # Forward and backward
            self.model.zero_grad()
            outputs = self.model(inputs)
            
            # Compute loss
            if targets is not None:
                loss = loss_fn(outputs.view(-1, outputs.size(-1)) if outputs.dim() == 3 else outputs, 
                              targets.view(-1) if targets.dim() > 1 else targets)
            else:
                # Self-supervised: use output as target
                if outputs.dim() == 3:
                    loss = loss_fn(
                        outputs[:, :-1].reshape(-1, outputs.size(-1)),
                        outputs[:, 1:].argmax(-1).reshape(-1)
                    )
                else:
                    loss = outputs.sum()  # Fallback
            
            loss.backward()
            
            # Update Fisher estimates
            self.update(inputs, targets, loss)
            
            samples_seen += inputs.size(0)
        
        # Finalize and return
        return self._finalize()
    
    def _reset(self):
        """Reset accumulated statistics."""
        self._fisher_cache.clear()
        self._sample_count = 0
    
    @abstractmethod
    def _finalize(self) -> Dict[str, torch.Tensor]:
        """Finalize computation and return results."""
        pass
    
    def get_layer_modules(self) -> Iterator[Tuple[str, nn.Module]]:
        """Iterate over adaptable layer modules."""
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
                yield name, module


class EmpiricalFisherEstimator(FisherEstimator):
    """
    Empirical Fisher using gradient outer products.
    
    Computes: F ≈ (1/N) Σ_i g_i g_i^T
    where g_i = ∇θ L(x_i, y_i)
    
    This is the most common approximation but requires O(d²) storage
    for full Fisher. We use diagonal approximation for efficiency.
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        damping: float = 1e-4,
        full_matrix: bool = False
    ):
        super().__init__(model, device, damping)
        self.full_matrix = full_matrix
        self._gradient_accum: Dict[str, torch.Tensor] = {}
    
    def update(
        self,
        inputs: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        loss: Optional[torch.Tensor] = None
    ) -> None:
        """Accumulate gradient statistics."""
        batch_size = inputs.size(0)
        
        for name, module in self.get_layer_modules():
            if module.weight.grad is None:
                continue
            
            grad = module.weight.grad.detach()
            
            if self.full_matrix:
                # Full outer product (expensive)
                grad_flat = grad.view(-1)
                outer = torch.outer(grad_flat, grad_flat)
                
                if name not in self._gradient_accum:
                    self._gradient_accum[name] = outer
                else:
                    self._gradient_accum[name] += outer
            else:
                # Diagonal approximation
                grad_sq = grad.view(-1) ** 2
                
                if name not in self._gradient_accum:
                    self._gradient_accum[name] = grad_sq
                else:
                    self._gradient_accum[name] += grad_sq
        
        self._sample_count += batch_size
    
    def get_fisher_block(self, layer_name: str) -> Optional[torch.Tensor]:
        """Get Fisher block for a layer."""
        if layer_name in self._fisher_cache:
            return self._fisher_cache[layer_name]
        return None
    
    def _finalize(self) -> Dict[str, torch.Tensor]:
        """Normalize and apply damping."""
        for name, accum in self._gradient_accum.items():
            fisher = accum / max(self._sample_count, 1)
            fisher = fisher + self.damping  # Add damping
            self._fisher_cache[name] = fisher
        
        return self._fisher_cache


class DiagonalFisherEstimator(EmpiricalFisherEstimator):
    """
    Diagonal Fisher approximation.
    
    Only stores diagonal elements, reducing storage from O(d²) to O(d).
    This is the default for FLOWRA as it balances accuracy and efficiency.
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        damping: float = 1e-4
    ):
        super().__init__(model, device, damping, full_matrix=False)


class KFACEstimator(FisherEstimator):
    """
    Kronecker-Factored Approximate Curvature (KFAC).
    
    Approximates the Fisher as: F ≈ A ⊗ G
    where:
    - A = E[a_i a_i^T] is the input covariance
    - G = E[g_i g_i^T] is the gradient covariance
    
    This provides a better approximation than diagonal while being
    more efficient than full Fisher.
    
    References:
        Martens & Grosse, 2015. "Optimizing Neural Networks with 
        Kronecker-factored Approximate Curvature"
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        damping: float = 1e-4,
        ema_decay: float = 0.95
    ):
        super().__init__(model, device, damping)
        self.ema_decay = ema_decay
        
        # KFAC factors: A (input covariance), G (gradient covariance)
        self._A_factors: Dict[str, torch.Tensor] = {}
        self._G_factors: Dict[str, torch.Tensor] = {}
        
        # Hooks for capturing activations
        self._hooks = []
        self._activations: Dict[str, torch.Tensor] = {}
        self._setup_hooks()
    
    def _setup_hooks(self):
        """Register forward hooks to capture activations."""
        def save_activation(name):
            def hook(module, input, output):
                if isinstance(input, tuple):
                    self._activations[name] = input[0].detach()
                else:
                    self._activations[name] = input.detach()
            return hook
        
        for name, module in self.get_layer_modules():
            hook = module.register_forward_hook(save_activation(name))
            self._hooks.append(hook)
    
    def _remove_hooks(self):
        """Remove all hooks."""
        for hook in self._hooks:
            hook.remove()
        self._hooks = []
    
    def update(
        self,
        inputs: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        loss: Optional[torch.Tensor] = None
    ) -> None:
        """Update KFAC factors."""
        for name, module in self.get_layer_modules():
            if module.weight.grad is None or name not in self._activations:
                continue
            
            # Input covariance factor A
            activation = self._activations[name]
            if activation.dim() > 2:
                activation = activation.view(activation.size(0), -1)
            
            # Add homogeneous coordinate for bias
            if module.bias is not None:
                ones = torch.ones(activation.size(0), 1, device=activation.device)
                activation = torch.cat([activation, ones], dim=1)
            
            A = torch.mm(activation.t(), activation) / activation.size(0)
            
            # Gradient covariance factor G
            grad = module.weight.grad.detach()
            if grad.dim() > 2:
                grad = grad.view(grad.size(0), -1)
            
            G = torch.mm(grad, grad.t()) / grad.size(0)
            
            # Update with EMA
            if name not in self._A_factors:
                self._A_factors[name] = A
                self._G_factors[name] = G
            else:
                self._A_factors[name] = (
                    self.ema_decay * self._A_factors[name] + 
                    (1 - self.ema_decay) * A
                )
                self._G_factors[name] = (
                    self.ema_decay * self._G_factors[name] + 
                    (1 - self.ema_decay) * G
                )
        
        self._sample_count += inputs.size(0)
    
    def get_fisher_block(self, layer_name: str) -> Optional[torch.Tensor]:
        """
        Get Fisher block (or its diagonal approximation).
        
        For KFAC, we return the diagonal of A ⊗ G.
        """
        if layer_name not in self._A_factors:
            return None
        
        A = self._A_factors[layer_name]
        G = self._G_factors[layer_name]
        
        # Diagonal of Kronecker product
        diag_A = torch.diag(A)
        diag_G = torch.diag(G)
        
        # Kronecker diagonal: diag(A ⊗ G) = diag(A) ⊗ diag(G)
        fisher_diag = torch.kron(diag_G, diag_A)
        
        return fisher_diag + self.damping
    
    def get_kfac_factors(self, layer_name: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get raw KFAC factors A and G for a layer."""
        return self._A_factors.get(layer_name), self._G_factors.get(layer_name)
    
    def _finalize(self) -> Dict[str, torch.Tensor]:
        """Compute final Fisher blocks."""
        for name in self._A_factors:
            self._fisher_cache[name] = self.get_fisher_block(name)
        
        self._remove_hooks()
        return self._fisher_cache
    
    def __del__(self):
        """Clean up hooks on deletion."""
        self._remove_hooks()


class BlockwiseFisherEstimator(FisherEstimator):
    """
    Block-wise Fisher estimation for attention layers.
    
    Computes separate Fisher blocks for Q, K, V projections
    to enable attention-specific analysis.
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        damping: float = 1e-4,
        attention_patterns: List[str] = None
    ):
        super().__init__(model, device, damping)
        self.attention_patterns = attention_patterns or [
            "query", "key", "value", "q_proj", "k_proj", "v_proj"
        ]
        self._diagonal_estimator = DiagonalFisherEstimator(model, device, damping)
        self._attention_fisher: Dict[str, Dict[str, torch.Tensor]] = {}
    
    def _is_attention_component(self, name: str) -> Optional[str]:
        """Check if layer is attention component, return type."""
        name_lower = name.lower()
        for pattern in self.attention_patterns:
            if pattern in name_lower:
                if "query" in pattern or "q_" in pattern:
                    return "query"
                elif "key" in pattern or "k_" in pattern:
                    return "key"
                elif "value" in pattern or "v_" in pattern:
                    return "value"
                return pattern
        return None
    
    def update(
        self,
        inputs: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        loss: Optional[torch.Tensor] = None
    ) -> None:
        """Update block-wise Fisher estimates."""
        self._diagonal_estimator.update(inputs, targets, loss)
        self._sample_count = self._diagonal_estimator._sample_count
    
    def get_fisher_block(self, layer_name: str) -> Optional[torch.Tensor]:
        """Get Fisher block."""
        return self._diagonal_estimator.get_fisher_block(layer_name)
    
    def _finalize(self) -> Dict[str, torch.Tensor]:
        """Finalize and organize by attention components."""
        base_fisher = self._diagonal_estimator._finalize()
        
        # Group attention components
        for name, fisher in base_fisher.items():
            attn_type = self._is_attention_component(name)
            if attn_type:
                # Extract attention block name
                parts = name.split('.')
                attn_block = '.'.join(parts[:-1])
                
                if attn_block not in self._attention_fisher:
                    self._attention_fisher[attn_block] = {}
                self._attention_fisher[attn_block][attn_type] = fisher
        
        return base_fisher
    
    def get_attention_fisher(self, attn_block_name: str) -> Dict[str, torch.Tensor]:
        """Get separate Fisher blocks for Q, K, V of an attention block."""
        return self._attention_fisher.get(attn_block_name, {})
