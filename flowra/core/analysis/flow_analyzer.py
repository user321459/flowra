"""
Main Flow Analyzer

Orchestrates Fisher Information, Jacobian, and redundancy analysis
to compute unified flow metrics for each layer.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Any
import numpy as np

from flowra.core.analysis.flow_metrics import FlowProfile, LayerFlowInfo, ModelFlowAnalysis
from flowra.core.analysis.fisher import (
    FisherEstimator,
    DiagonalFisherEstimator,
    KFACEstimator,
    BlockwiseFisherEstimator,
)
from flowra.core.analysis.jacobian import (
    JacobianEstimator,
    PowerIterationEstimator,
    AnalyticalJacobianEstimator,
)
from flowra.core.analysis.redundancy import RedundancyAnalyzer


class FlowAnalyzer:
    """
    Main class for flow analysis.
    
    Computes flow metrics (ψ, γ, ρ) for each layer based on:
    - Fisher Information (gradient sensitivity)
    - Jacobian norms (output influence)
    - Singular value analysis (redundancy)
    
    Example:
        >>> analyzer = FlowAnalyzer(model)
        >>> analysis = analyzer.analyze(calibration_loader)
        >>> print(analysis.summary())
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: torch.device = None,
        fisher_type: str = "diagonal",
        num_fisher_samples: int = 100,
        jacobian_iterations: int = 10,
        attention_patterns: List[str] = None
    ):
        self.model = model
        self.device = device or torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
        self.model.to(self.device)
        
        self.num_fisher_samples = num_fisher_samples
        self.attention_patterns = attention_patterns or [
            "query", "key", "value", "q_proj", "k_proj", "v_proj", "qkv"
        ]
        
        # Initialize estimators
        if fisher_type == "diagonal":
            self.fisher_estimator = DiagonalFisherEstimator(model, self.device)
        elif fisher_type == "kfac":
            self.fisher_estimator = KFACEstimator(model, self.device)
        elif fisher_type == "blockwise":
            self.fisher_estimator = BlockwiseFisherEstimator(
                model, self.device, attention_patterns=self.attention_patterns
            )
        else:
            self.fisher_estimator = DiagonalFisherEstimator(model, self.device)
        
        self.jacobian_estimator = PowerIterationEstimator(
            model, self.device, num_iterations=jacobian_iterations
        )
        
        self.redundancy_analyzer = RedundancyAnalyzer(model, self.device)
        
        # Results cache
        self._fisher_blocks: Dict[str, torch.Tensor] = {}
        self._jacobian_norms: Dict[str, float] = {}
        self._redundancies: Dict[str, float] = {}
        self._flow_profiles: Dict[str, FlowProfile] = {}
        
        # Identify adaptable layers
        self.adaptable_layers = self._identify_adaptable_layers()
    
    def _identify_adaptable_layers(self) -> Dict[str, Dict[str, Any]]:
        """Identify layers that can be adapted."""
        adaptable = {}
        
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                is_attention = any(
                    p in name.lower() for p in self.attention_patterns
                )
                adaptable[name] = {
                    'module': module,
                    'type': 'attention' if is_attention else 'linear',
                    'shape': module.weight.shape,
                    'params': module.weight.numel(),
                }
            elif isinstance(module, (nn.Conv1d, nn.Conv2d)):
                adaptable[name] = {
                    'module': module,
                    'type': 'conv',
                    'shape': module.weight.shape,
                    'params': module.weight.numel(),
                }
            elif isinstance(module, (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d)):
                adaptable[name] = {
                    'module': module,
                    'type': 'norm',
                    'shape': (module.weight.shape[0],) if hasattr(module, 'weight') else (0,),
                    'params': module.weight.numel() if hasattr(module, 'weight') else 0,
                }
        
        return adaptable
    
    def analyze(
        self,
        dataloader: DataLoader,
        loss_fn: nn.Module = None
    ) -> ModelFlowAnalysis:
        """
        Perform complete flow analysis.
        
        Args:
            dataloader: Calibration data
            loss_fn: Loss function (default: CrossEntropyLoss)
        
        Returns:
            ModelFlowAnalysis with all layer information
        """
        if loss_fn is None:
            loss_fn = nn.CrossEntropyLoss()
        
        print("Computing Fisher Information...")
        self._fisher_blocks = self.fisher_estimator.compute(
            dataloader, loss_fn, num_samples=self.num_fisher_samples
        )
        
        print("Computing Jacobian norms...")
        self._jacobian_norms = self.jacobian_estimator.estimate(dataloader)
        jacobian_normalized = self.jacobian_estimator.get_normalized_norms()
        
        print("Analyzing redundancy...")
        self._redundancies = self.redundancy_analyzer.analyze()
        
        print("Computing flow profiles...")
        analysis = ModelFlowAnalysis()
        
        # Compute total Fisher trace for normalization
        total_fisher_trace = sum(
            f.sum().item() for f in self._fisher_blocks.values()
        )
        
        for name, info in self.adaptable_layers.items():
            # Flow Sensitivity (ψ)
            if name in self._fisher_blocks:
                fisher_trace = self._fisher_blocks[name].sum().item()
                if total_fisher_trace > 0:
                    psi = fisher_trace / total_fisher_trace
                else:
                    psi = 1.0 / len(self.adaptable_layers)
            else:
                psi = 0.0
            
            # Flow Conductance (γ)
            if name in self._fisher_blocks and name in jacobian_normalized:
                fisher = self._fisher_blocks[name]
                lambda_max = fisher.max().item()
                lambda_min = fisher.min().item()
                
                if lambda_max > 0:
                    spectral_ratio = (lambda_max - lambda_min) / lambda_max
                else:
                    spectral_ratio = 0.0
                
                gamma = spectral_ratio * jacobian_normalized.get(name, 0.0)
            else:
                gamma = 0.0
            
            # Redundancy (ρ)
            rho = self._redundancies.get(name, 0.0)
            
            # Create flow profile
            flow_profile = FlowProfile(
                psi=psi,
                gamma=gamma,
                rho=rho,
                fisher_trace=fisher_trace if name in self._fisher_blocks else None,
                jacobian_norm=self._jacobian_norms.get(name),
                effective_rank=self.redundancy_analyzer.get_effective_rank(name),
            )
            
            self._flow_profiles[name] = flow_profile
            
            # Create layer info
            layer_info = LayerFlowInfo(
                name=name,
                flow_profile=flow_profile,
                layer_type=info['type'],
                weight_shape=info['shape'],
                fisher_block=self._fisher_blocks.get(name),
                num_parameters=info['params'],
                is_attention_component=info['type'] == 'attention',
            )
            
            analysis.layer_info[name] = layer_info
        
        # Compute global statistics
        analysis.total_parameters = sum(
            info['params'] for info in self.adaptable_layers.values()
        )
        analysis.total_fisher_trace = total_fisher_trace
        
        # Compute rankings
        analysis._compute_rankings()
        
        print(f"Analysis complete. Analyzed {len(analysis.layer_info)} layers.")
        
        return analysis
    
    def get_flow_profiles(self) -> Dict[str, FlowProfile]:
        """Return cached flow profiles."""
        return self._flow_profiles
    
    def get_conductances(self) -> Dict[str, float]:
        """Return flow conductances for rank allocation."""
        return {
            name: profile.gamma 
            for name, profile in self._flow_profiles.items()
        }
    
    def get_sensitivities(self) -> Dict[str, float]:
        """Return flow sensitivities."""
        return {
            name: profile.psi 
            for name, profile in self._flow_profiles.items()
        }
