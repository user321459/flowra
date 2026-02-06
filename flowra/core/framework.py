"""
Main FLOWRA Framework Class

Integrates all components through a hierarchical process:
Phase 1: Flow Analysis
Phase 2: Architecture Configuration
Phase 3: Dynamic Training
Phase 4: Composition
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Any
import copy

from flowra.configs import FlowConfig
from flowra.core.analysis import FlowAnalyzer, FlowProfile, LayerFlowInfo, ModelFlowAnalysis
from flowra.core.adapters import PolymorphicAdapterFactory, BaseAdapter
from flowra.core.training import FlowraTrainer


class FLOWRA:
    """
    FLOWRA: Flow-based Low-Rank Adaptation Framework
    
    Main class that orchestrates flow analysis, adapter creation,
    training, and multi-task composition.
    
    Example:
        >>> config = FlowConfig(total_budget=0.01)
        >>> framework = FLOWRA(model, config)
        >>> adapted = framework.adapt(calibration_loader)
        >>> framework.train(train_loader, epochs=3)
    """
    
    def __init__(self, model: nn.Module, config: FlowConfig = None):
        self.config = config or FlowConfig()
        self.base_model = model
        self.device = torch.device(self.config.device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        self.base_model.to(self.device)
        
        # Initialize components
        self.flow_analyzer = FlowAnalyzer(model, self.device)
        self.adapter_factory = PolymorphicAdapterFactory()
        
        # State
        self.analysis: Optional[ModelFlowAnalysis] = None
        self.rank_allocation: Dict[str, int] = {}
        self.adapters: Dict[str, BaseAdapter] = {}
        self.adapted_model: Optional[nn.Module] = None
        
        # Compute budget
        self.base_params = sum(p.numel() for p in model.parameters())
        self.budget = int(self.base_params * self.config.total_budget)
        
        if self.config.verbose:
            print(f"FLOWRA initialized: {self.base_params:,} base params, "
                  f"{self.budget:,} adaptation budget")
    
    def analyze(self, dataloader: DataLoader, loss_fn: nn.Module = None) -> ModelFlowAnalysis:
        """Phase 1: Analyze information flow."""
        if self.config.verbose:
            print("\n" + "="*60)
            print("Phase 1: Flow Analysis")
            print("="*60)
        
        self.analysis = self.flow_analyzer.analyze(dataloader, loss_fn)
        return self.analysis
    
    def allocate_ranks(self) -> Dict[str, int]:
        """Phase 2: Allocate ranks based on flow analysis."""
        if self.analysis is None:
            raise RuntimeError("Run analyze() first")
        
        if self.config.verbose:
            print("\n" + "="*60)
            print("Phase 2: Rank Allocation")
            print("="*60)
        
        # Import allocator
        from flowra.algorithms.allocation import SpectralRankAllocator
        allocator = SpectralRankAllocator(
            min_rank=self.config.min_rank,
            max_rank_ratio=self.config.max_rank_ratio
        )
        
        conductances = {name: info.flow_profile.gamma for name, info in self.analysis.layer_info.items()}
        shapes = {name: info.weight_shape for name, info in self.analysis.layer_info.items()}
        
        self.rank_allocation = allocator.allocate(conductances, shapes, self.budget)
        
        if self.config.verbose:
            total = sum(self.rank_allocation.values())
            print(f"Allocated ranks to {len(self.rank_allocation)} layers, total rank: {total}")
        
        return self.rank_allocation
    
    def create_adapters(self) -> Dict[str, BaseAdapter]:
        """Create adapter modules based on allocation."""
        if not self.rank_allocation:
            raise RuntimeError("Run allocate_ranks() first")
        
        if self.config.verbose:
            print("\nCreating adapters...")
        
        self.adapted_model = copy.deepcopy(self.base_model)
        
        for name, rank in self.rank_allocation.items():
            # Get layer
            parts = name.split('.')
            parent = self.adapted_model
            for part in parts[:-1]:
                parent = getattr(parent, part)
            layer = getattr(parent, parts[-1])
            
            # Get flow profile
            profile = self.analysis.layer_info[name].flow_profile if self.analysis else None
            scaling = (profile.psi / max(rank, 1)) ** 0.5 if profile and profile.psi > 0 else 0.01
            
            # Create adapter
            adapter = self.adapter_factory.create_adapter(
                layer=layer,
                layer_name=name,
                rank=rank,
                flow_profile=profile,
                scaling=scaling,
                dropout=self.config.dropout
            )
            
            # Replace in model
            setattr(parent, parts[-1], adapter)
            self.adapters[name] = adapter
        
        if self.config.verbose:
            trainable = sum(a.get_trainable_params() for a in self.adapters.values())
            print(f"Created {len(self.adapters)} adapters with {trainable:,} trainable params")
        
        return self.adapters
    
    def adapt(self, calibration_data: DataLoader, loss_fn: nn.Module = None) -> nn.Module:
        """Complete adaptation: analyze + allocate + create adapters."""
        self.analyze(calibration_data, loss_fn)
        self.allocate_ranks()
        self.create_adapters()
        return self.adapted_model
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader = None,
        epochs: int = 3,
        lr: float = 1e-4,
        **kwargs
    ):
        """Phase 3: Train the adapted model."""
        if self.adapted_model is None:
            raise RuntimeError("Run adapt() first")
        
        if self.config.verbose:
            print("\n" + "="*60)
            print("Phase 3: Training")
            print("="*60)
        
        trainer = FlowraTrainer(
            model=self.adapted_model,
            adapters=self.adapters,
            train_loader=train_loader,
            val_loader=val_loader,
            device=self.device
        )
        
        # Setup optimizer
        params = [p for a in self.adapters.values() for p in a.parameters() if p.requires_grad]
        trainer.optimizer = torch.optim.AdamW(params, lr=lr)
        
        total_steps = len(train_loader) * epochs
        trainer.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(trainer.optimizer, total_steps)
        
        trainer.train(epochs=epochs, **kwargs)
        
        return trainer.training_history
    
    def merge_into_base(self) -> nn.Module:
        """Merge adapters into base weights for efficient inference."""
        merged = copy.deepcopy(self.base_model)
        
        for name, adapter in self.adapters.items():
            parts = name.split('.')
            parent = merged
            for part in parts[:-1]:
                parent = getattr(parent, part)
            
            layer = getattr(parent, parts[-1])
            adaptation = adapter.get_adaptation()
            
            with torch.no_grad():
                layer.weight.add_(adapter.scaling * adaptation)
        
        return merged
    
    def save_adapters(self, path: str):
        """Save adapter state."""
        state = {name: a.state_dict() for name, a in self.adapters.items()}
        state['config'] = self.config.to_dict()
        state['rank_allocation'] = self.rank_allocation
        torch.save(state, path)
    
    def load_adapters(self, path: str):
        """Load adapter state."""
        state = torch.load(path, map_location=self.device)
        for name, adapter_state in state.items():
            if name in self.adapters:
                self.adapters[name].load_state_dict(adapter_state)
    
    def summary(self) -> str:
        """Return summary string."""
        lines = [
            "FLOWRA Summary",
            "="*40,
            f"Base parameters: {self.base_params:,}",
            f"Budget: {self.config.total_budget*100:.2f}%",
        ]
        if self.adapters:
            trainable = sum(a.get_trainable_params() for a in self.adapters.values())
            lines.append(f"Adapted layers: {len(self.adapters)}")
            lines.append(f"Trainable parameters: {trainable:,}")
        return "\n".join(lines)
