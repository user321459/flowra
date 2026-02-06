"""
Functions for computing and analyzing PEFT-specific metrics.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import warnings


# ============================================================================
# Parameter Efficiency Metrics
# ============================================================================

def compute_trainable_params(
    total_params: int,
    trainable_params: int,
) -> Dict[str, float]:
    """
    Compute parameter efficiency metrics.
    
    Parameters
    ----------
    total_params : int
        Total model parameters
    trainable_params : int
        Number of trainable parameters
        
    Returns
    -------
    Dict[str, float]
        Dictionary with parameter metrics
    """
    frozen_params = total_params - trainable_params
    
    return {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'frozen_params': frozen_params,
        'trainable_percent': trainable_params / total_params * 100,
        'frozen_percent': frozen_params / total_params * 100,
        'compression_ratio': total_params / trainable_params if trainable_params > 0 else float('inf'),
    }


def compute_lora_params(
    in_features: int,
    out_features: int,
    rank: int,
    num_layers: int = 1,
    include_bias: bool = False,
) -> int:
    """
    Compute number of parameters in LoRA adaptation.
    
    Parameters
    ----------
    in_features : int
        Input dimension
    out_features : int
        Output dimension
    rank : int
        LoRA rank
    num_layers : int
        Number of adapted layers
    include_bias : bool
        Whether to include bias parameters
        
    Returns
    -------
    int
        Total LoRA parameters
    """
    # A: in_features x rank, B: rank x out_features
    lora_params = (in_features * rank + rank * out_features) * num_layers
    
    if include_bias:
        lora_params += out_features * num_layers
    
    return lora_params


def compute_adapter_params(
    hidden_size: int,
    bottleneck_size: int,
    num_layers: int = 1,
) -> int:
    """
    Compute number of parameters in adapter layers.
    
    Parameters
    ----------
    hidden_size : int
        Model hidden dimension
    bottleneck_size : int
        Adapter bottleneck dimension
    num_layers : int
        Number of adapter layers
        
    Returns
    -------
    int
        Total adapter parameters
    """
    # Down: hidden x bottleneck, Up: bottleneck x hidden
    # Plus biases
    params_per_adapter = (hidden_size * bottleneck_size + bottleneck_size +
                          bottleneck_size * hidden_size + hidden_size)
    return params_per_adapter * num_layers


def estimate_flowra_params(
    base_lora_params: int,
    avg_rank_ratio: float = 0.7,
    overhead_ratio: float = 0.05,
) -> int:
    """
    Estimate FLOWRA parameter count.
    
    Parameters
    ----------
    base_lora_params : int
        Equivalent LoRA parameters (uniform rank)
    avg_rank_ratio : float
        Average rank as ratio of base rank (due to dynamic allocation)
    overhead_ratio : float
        Additional overhead for flow tracking
        
    Returns
    -------
    int
        Estimated FLOWRA parameters
    """
    effective_params = base_lora_params * avg_rank_ratio
    overhead = base_lora_params * overhead_ratio
    return int(effective_params + overhead)


# ============================================================================
# Rank Allocation Metrics
# ============================================================================

@dataclass
class RankAllocation:
    """Container for rank allocation across layers."""
    layer_names: List[str]
    initial_ranks: List[int]
    final_ranks: List[float]
    
    @property
    def reduction_percent(self) -> List[float]:
        return [(i - f) / i * 100 for i, f in zip(self.initial_ranks, self.final_ranks)]
    
    @property
    def avg_rank(self) -> float:
        return np.mean(self.final_ranks)
    
    @property
    def total_reduction(self) -> float:
        return (sum(self.initial_ranks) - sum(self.final_ranks)) / sum(self.initial_ranks) * 100
    
    def to_dict(self) -> Dict:
        return {
            'layers': self.layer_names,
            'initial': self.initial_ranks,
            'final': self.final_ranks,
            'reduction': self.reduction_percent,
            'avg_rank': self.avg_rank,
            'total_reduction': self.total_reduction,
        }


def compute_effective_rank(
    singular_values: np.ndarray,
    threshold: float = 0.01,
) -> int:
    """
    Compute effective rank based on singular value decay.
    
    Parameters
    ----------
    singular_values : np.ndarray
        Singular values (sorted descending)
    threshold : float
        Threshold relative to largest singular value
        
    Returns
    -------
    int
        Effective rank
    """
    if len(singular_values) == 0:
        return 0
    
    sv_normalized = singular_values / singular_values[0]
    return int(np.sum(sv_normalized > threshold))


def compute_rank_entropy(
    ranks: List[float],
) -> float:
    """
    Compute entropy of rank distribution (uniformity measure).
    
    Parameters
    ----------
    ranks : List[float]
        Rank values across layers
        
    Returns
    -------
    float
        Entropy value (higher = more uniform)
    """
    ranks = np.array(ranks)
    total = np.sum(ranks)
    if total == 0:
        return 0.0
    
    probs = ranks / total
    probs = probs[probs > 0] 
    return -np.sum(probs * np.log2(probs))


def analyze_rank_dynamics(
    rank_history: List[List[float]],
    layer_names: Optional[List[str]] = None,
) -> Dict[str, np.ndarray]:
    """
    Analyze rank dynamics during training.
    
    Parameters
    ----------
    rank_history : List[List[float]]
        Rank values at each checkpoint [checkpoint][layer]
    layer_names : List[str], optional
        Names of layers
        
    Returns
    -------
    Dict[str, np.ndarray]
        Analysis results
    """
    history = np.array(rank_history)
    n_checkpoints, n_layers = history.shape
    
    if layer_names is None:
        layer_names = [f'Layer_{i}' for i in range(n_layers)]
    
    # Compute trends
    trends = np.diff(history, axis=0)  # Change between checkpoints
    
    # Classify layers
    final_vs_initial = history[-1] - history[0]
    increasing = final_vs_initial > 0.5
    decreasing = final_vs_initial < -0.5
    stable = ~increasing & ~decreasing
    
    return {
        'history': history,
        'trends': trends,
        'final_vs_initial': final_vs_initial,
        'increasing_layers': np.array(layer_names)[increasing],
        'decreasing_layers': np.array(layer_names)[decreasing],
        'stable_layers': np.array(layer_names)[stable],
        'volatility': np.std(trends, axis=0),  # Per-layer volatility
    }


# ============================================================================
# Flow Sensitivity Metrics
# ============================================================================

def compute_flow_sensitivity(
    gradients: np.ndarray,
    activations: np.ndarray,
    method: str = 'fisher',
) -> float:
    """
    Compute information flow sensitivity for a layer.
    
    Parameters
    ----------
    gradients : np.ndarray
        Gradient values
    activations : np.ndarray
        Activation values
    method : str
        Sensitivity method: 'fisher', 'magnitude', 'snr'
        
    Returns
    -------
    float
        Sensitivity score
    """
    if method == 'fisher':
        # Fisher information approximation
        return np.mean(gradients ** 2)
    
    elif method == 'magnitude':
        # Simple gradient magnitude
        return np.mean(np.abs(gradients))
    
    elif method == 'snr':
        # Signal-to-noise ratio
        signal = np.mean(gradients)
        noise = np.std(gradients)
        return abs(signal) / noise if noise > 0 else 0.0
    
    else:
        raise ValueError(f"Unknown sensitivity method: {method}")


def compute_layer_importance(
    layer_sensitivities: Dict[str, float],
    normalization: str = 'softmax',
) -> Dict[str, float]:
    """
    Normalize layer sensitivities to importance scores.
    
    Parameters
    ----------
    layer_sensitivities : Dict[str, float]
        Layer name -> sensitivity mapping
    normalization : str
        Normalization method: 'softmax', 'linear', 'rank'
        
    Returns
    -------
    Dict[str, float]
        Layer importance scores (sum to 1)
    """
    names = list(layer_sensitivities.keys())
    values = np.array(list(layer_sensitivities.values()))
    
    if normalization == 'softmax':
        exp_values = np.exp(values - np.max(values)) 
        importance = exp_values / np.sum(exp_values)
    
    elif normalization == 'linear':
        total = np.sum(values)
        importance = values / total if total > 0 else np.ones_like(values) / len(values)
    
    elif normalization == 'rank':
        ranks = np.argsort(np.argsort(values)) + 1  # Rank from 1
        importance = ranks / np.sum(ranks)
    
    else:
        raise ValueError(f"Unknown normalization: {normalization}")
    
    return dict(zip(names, importance))


# ============================================================================
# Quality Retention Metrics
# ============================================================================

def compute_quality_retention(
    peft_score: float,
    full_ft_score: float,
) -> float:
    """
    Compute quality retention as percentage of full fine-tuning.
    
    Parameters
    ----------
    peft_score : float
        PEFT method score
    full_ft_score : float
        Full fine-tuning score
        
    Returns
    -------
    float
        Quality retention percentage
    """
    if full_ft_score == 0:
        return 0.0
    return peft_score / full_ft_score * 100


def compute_efficiency_frontier_distance(
    quality: float,
    params_percent: float,
    reference_quality: float = 100.0,
    reference_params: float = 0.0,
) -> float:
    """
    Compute distance to ideal efficiency frontier point.
    
    Parameters
    ----------
    quality : float
        Quality score
    params_percent : float
        Parameter percentage
    reference_quality : float
        Ideal quality (default: 100)
    reference_params : float
        Ideal params (default: 0)
        
    Returns
    -------
    float
        Euclidean distance to ideal point (lower is better)
    """
    # Normalize to 0-1 scale
    quality_gap = (reference_quality - quality) / reference_quality
    params_gap = params_percent / 100
    
    return np.sqrt(quality_gap**2 + params_gap**2)


def compute_gap_reduction(
    method_gap: float,
    baseline_gap: float,
) -> float:
    """
    Compute how much a method reduces the gap to full FT.
    
    Parameters
    ----------
    method_gap : float
        Gap for the method (negative value)
    baseline_gap : float
        Gap for baseline (negative value)
        
    Returns
    -------
    float
        Gap reduction percentage
    """
    if baseline_gap == 0:
        return 0.0
    return (baseline_gap - method_gap) / abs(baseline_gap) * 100


# ============================================================================
# Multi-Task Metrics
# ============================================================================

def compute_task_interference(
    individual_scores: Dict[str, float],
    joint_scores: Dict[str, float],
) -> Dict[str, float]:
    """
    Compute task interference in multi-task setting.
    
    Parameters
    ----------
    individual_scores : Dict[str, float]
        Scores when training on each task individually
    joint_scores : Dict[str, float]
        Scores when training jointly on all tasks
        
    Returns
    -------
    Dict[str, float]
        Interference scores (negative = interference)
    """
    interference = {}
    for task in individual_scores:
        if task in joint_scores:
            interference[task] = joint_scores[task] - individual_scores[task]
    return interference


def compute_forgetting(
    initial_scores: Dict[str, float],
    final_scores: Dict[str, float],
) -> Dict[str, float]:
    """
    Compute catastrophic forgetting metrics.
    
    Parameters
    ----------
    initial_scores : Dict[str, float]
        Scores after learning each task
    final_scores : Dict[str, float]
        Scores after learning all tasks
        
    Returns
    -------
    Dict[str, float]
        Forgetting scores (negative = forgetting)
    """
    forgetting = {}
    for task in initial_scores:
        if task in final_scores:
            forgetting[task] = final_scores[task] - initial_scores[task]
    return forgetting


def compute_backward_transfer(
    score_matrix: np.ndarray,
) -> float:
    """
    Compute backward transfer metric.
    
    Parameters
    ----------
    score_matrix : np.ndarray
        Matrix where [i,j] = score on task j after learning task i
        
    Returns
    -------
    float
        Average backward transfer
    """
    n_tasks = score_matrix.shape[0]
    if n_tasks < 2:
        return 0.0
    
    bt_sum = 0
    count = 0
    
    for j in range(n_tasks - 1):
        for i in range(j + 1, n_tasks):
            bt_sum += score_matrix[i, j] - score_matrix[j, j]
            count += 1
    
    return bt_sum / count if count > 0 else 0.0


# ============================================================================
# Convergence Metrics
# ============================================================================

def compute_convergence_speed(
    loss_curve: List[float],
    target_loss: Optional[float] = None,
    threshold_ratio: float = 0.95,
) -> Dict[str, float]:
    """
    Compute convergence speed metrics.
    
    Parameters
    ----------
    loss_curve : List[float]
        Training loss values
    target_loss : float, optional
        Target loss value
    threshold_ratio : float
        Ratio of final loss to consider converged
        
    Returns
    -------
    Dict[str, float]
        Convergence metrics
    """
    losses = np.array(loss_curve)
    
    if target_loss is None:
        target_loss = losses[-1] * (1 + (1 - threshold_ratio))
    
    # Find first step where loss is below threshold
    converged_idx = np.where(losses <= target_loss)[0]
    steps_to_converge = converged_idx[0] if len(converged_idx) > 0 else len(losses)
    
    # Compute area under curve
    auc = np.trapz(losses)
    
    # Compute average loss
    avg_loss = np.mean(losses)
    
    return {
        'steps_to_converge': steps_to_converge,
        'final_loss': losses[-1],
        'auc': auc,
        'avg_loss': avg_loss,
        'convergence_ratio': steps_to_converge / len(losses),
    }


def compute_gradient_alignment(
    approx_gradient: np.ndarray,
    true_gradient: np.ndarray,
) -> float:
    """
    Compute alignment between approximate and true gradients.
    
    Parameters
    ----------
    approx_gradient : np.ndarray
        Approximate gradient (e.g., from PEFT)
    true_gradient : np.ndarray
        True full gradient
        
    Returns
    -------
    float
        Cosine similarity (alignment score)
    """
    approx_flat = approx_gradient.flatten()
    true_flat = true_gradient.flatten()
    
    norm_approx = np.linalg.norm(approx_flat)
    norm_true = np.linalg.norm(true_flat)
    
    if norm_approx == 0 or norm_true == 0:
        return 0.0
    
    return np.dot(approx_flat, true_flat) / (norm_approx * norm_true)


# ============================================================================
# Subspace Analysis
# ============================================================================

def compute_subspace_overlap(
    basis1: np.ndarray,
    basis2: np.ndarray,
) -> float:
    """
    Compute overlap between two subspaces.
    
    Parameters
    ----------
    basis1 : np.ndarray
        Orthonormal basis for first subspace (columns)
    basis2 : np.ndarray
        Orthonormal basis for second subspace (columns)
        
    Returns
    -------
    float
        Subspace overlap (0-1)
    """
    # Compute principal angles via SVD
    M = basis1.T @ basis2
    singular_values = np.linalg.svd(M, compute_uv=False)
    
    # Overlap is sum of squared cosines of principal angles
    overlap = np.sum(singular_values ** 2) / min(basis1.shape[1], basis2.shape[1])
    return overlap


def analyze_weight_update_subspace(
    initial_weights: np.ndarray,
    final_weights: np.ndarray,
    rank: int = 16,
) -> Dict[str, float]:
    """
    Analyze the subspace of weight updates.
    
    Parameters
    ----------
    initial_weights : np.ndarray
        Initial weight matrix
    final_weights : np.ndarray
        Final weight matrix
    rank : int
        Rank for low-rank approximation
        
    Returns
    -------
    Dict[str, float]
        Subspace analysis metrics
    """
    delta = final_weights - initial_weights
    
    # SVD of weight update
    U, S, Vt = np.linalg.svd(delta, full_matrices=False)
    
    # Effective rank
    total_var = np.sum(S ** 2)
    cumsum = np.cumsum(S ** 2)
    effective_rank = np.searchsorted(cumsum / total_var, 0.99) + 1
    
    # Energy in top-k components
    top_k_energy = np.sum(S[:rank] ** 2) / total_var if total_var > 0 else 0
    
    # Condition number
    cond_number = S[0] / S[-1] if S[-1] > 0 else float('inf')
    
    return {
        'effective_rank': effective_rank,
        'top_k_energy': top_k_energy,
        'spectral_norm': S[0],
        'frobenius_norm': np.sqrt(total_var),
        'condition_number': cond_number,
        'singular_values': S[:rank].tolist(),
    }
