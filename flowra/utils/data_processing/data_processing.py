"""
Functions for loading, processing, and analyzing experimental results.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field
from pathlib import Path
import json
import warnings


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class ExperimentResult:
    """Container for a single experiment result."""
    method: str
    benchmark: str
    score: float
    params_percent: float
    training_time: Optional[float] = None
    memory_gb: Optional[float] = None
    config: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BenchmarkSuite:
    """Collection of benchmark results."""
    name: str
    results: List[ExperimentResult] = field(default_factory=list)
    
    def add_result(self, result: ExperimentResult) -> None:
        self.results.append(result)
    
    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([
            {
                'method': r.method,
                'benchmark': r.benchmark,
                'score': r.score,
                'params_percent': r.params_percent,
                'training_time': r.training_time,
                'memory_gb': r.memory_gb,
                **r.config,
                **r.metadata,
            }
            for r in self.results
        ])
    
    def get_method_scores(self, method: str) -> Dict[str, float]:
        return {r.benchmark: r.score for r in self.results if r.method == method}
    
    def get_benchmark_comparison(self, benchmark: str) -> Dict[str, float]:
        return {r.method: r.score for r in self.results if r.benchmark == benchmark}


# ============================================================================
# Normalization Functions
# ============================================================================

def normalize_scores(
    scores: Dict[str, float],
    reference: Optional[float] = None,
    method: str = 'percentage',
    invert: bool = False,
) -> Dict[str, float]:
    """
    Normalize scores for comparison.
    
    Parameters
    ----------
    scores : Dict[str, float]
        Method -> score mapping
    reference : float, optional
        Reference value (default: max score)
    method : str
        Normalization method: 'percentage', 'minmax', 'zscore'
    invert : bool
        Whether to invert (for metrics where lower is better)
        
    Returns
    -------
    Dict[str, float]
        Normalized scores
    """
    values = list(scores.values())
    
    if reference is None:
        reference = max(values)
    
    if method == 'percentage':
        if invert:
            normalized = {k: (reference - v) / reference * 100 for k, v in scores.items()}
        else:
            normalized = {k: v / reference * 100 for k, v in scores.items()}
    
    elif method == 'minmax':
        min_val, max_val = min(values), max(values)
        range_val = max_val - min_val
        if range_val == 0:
            normalized = {k: 50.0 for k in scores}
        else:
            normalized = {k: (v - min_val) / range_val * 100 for k, v in scores.items()}
            if invert:
                normalized = {k: 100 - v for k, v in normalized.items()}
    
    elif method == 'zscore':
        mean_val = np.mean(values)
        std_val = np.std(values)
        if std_val == 0:
            normalized = {k: 0.0 for k in scores}
        else:
            normalized = {k: (v - mean_val) / std_val for k, v in scores.items()}
            if invert:
                normalized = {k: -v for k, v in normalized.items()}
    
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    
    return normalized


def compute_gap_to_reference(
    scores: Dict[str, float],
    reference_method: str = 'Full FT',
) -> Dict[str, float]:
    """
    Compute performance gap relative to reference method.
    
    Parameters
    ----------
    scores : Dict[str, float]
        Method -> score mapping
    reference_method : str
        Name of reference method
        
    Returns
    -------
    Dict[str, float]
        Gap values (negative means worse than reference)
    """
    if reference_method not in scores:
        raise ValueError(f"Reference method '{reference_method}' not in scores")
    
    ref_score = scores[reference_method]
    return {k: v - ref_score for k, v in scores.items()}


def compute_relative_improvement(
    scores: Dict[str, float],
    baseline_method: str = 'LoRA',
) -> Dict[str, float]:
    """
    Compute relative improvement over baseline.
    
    Parameters
    ----------
    scores : Dict[str, float]
        Method -> score mapping
    baseline_method : str
        Name of baseline method
        
    Returns
    -------
    Dict[str, float]
        Relative improvement percentages
    """
    if baseline_method not in scores:
        raise ValueError(f"Baseline method '{baseline_method}' not in scores")
    
    baseline = scores[baseline_method]
    if baseline == 0:
        warnings.warn("Baseline score is 0, returning raw differences")
        return {k: v - baseline for k, v in scores.items()}
    
    return {k: (v - baseline) / baseline * 100 for k, v in scores.items()}


# ============================================================================
# Statistical Analysis
# ============================================================================

def compute_statistics(
    values: List[float],
) -> Dict[str, float]:
    """
    Compute basic statistics for a list of values.
    
    Parameters
    ----------
    values : List[float]
        Input values
        
    Returns
    -------
    Dict[str, float]
        Statistics dictionary
    """
    arr = np.array(values)
    return {
        'mean': np.mean(arr),
        'std': np.std(arr),
        'min': np.min(arr),
        'max': np.max(arr),
        'median': np.median(arr),
        'q25': np.percentile(arr, 25),
        'q75': np.percentile(arr, 75),
    }


def compute_effect_size(
    group1: List[float],
    group2: List[float],
    method: str = 'cohens_d',
) -> float:
    """
    Compute effect size between two groups.
    
    Parameters
    ----------
    group1 : List[float]
        First group values
    group2 : List[float]
        Second group values
    method : str
        Effect size method: 'cohens_d', 'hedges_g'
        
    Returns
    -------
    float
        Effect size value
    """
    arr1, arr2 = np.array(group1), np.array(group2)
    n1, n2 = len(arr1), len(arr2)
    
    mean_diff = np.mean(arr1) - np.mean(arr2)
    
    if method == 'cohens_d':
        pooled_std = np.sqrt(((n1-1)*np.var(arr1, ddof=1) + (n2-1)*np.var(arr2, ddof=1)) / (n1+n2-2))
        return mean_diff / pooled_std if pooled_std > 0 else 0.0
    
    elif method == 'hedges_g':
        pooled_std = np.sqrt(((n1-1)*np.var(arr1, ddof=1) + (n2-1)*np.var(arr2, ddof=1)) / (n1+n2-2))
        d = mean_diff / pooled_std if pooled_std > 0 else 0.0
        # Hedges' g correction
        correction = 1 - (3 / (4*(n1+n2) - 9))
        return d * correction
    
    else:
        raise ValueError(f"Unknown effect size method: {method}")


def rank_methods(
    results: Dict[str, Dict[str, float]],
    higher_is_better: Optional[Dict[str, bool]] = None,
) -> pd.DataFrame:
    """
    Rank methods across multiple benchmarks.
    
    Parameters
    ----------
    results : Dict[str, Dict[str, float]]
        Nested dict: method -> benchmark -> score
    higher_is_better : Dict[str, bool], optional
        Whether higher is better for each benchmark
        
    Returns
    -------
    pd.DataFrame
        Ranking dataframe
    """
    df = pd.DataFrame(results).T
    
    if higher_is_better is None:
        higher_is_better = {col: True for col in df.columns}
    
    ranks = pd.DataFrame(index=df.index)
    for col in df.columns:
        ascending = not higher_is_better.get(col, True)
        ranks[col] = df[col].rank(ascending=ascending)
    
    ranks['mean_rank'] = ranks.mean(axis=1)
    ranks['wins'] = (ranks.drop('mean_rank', axis=1) == 1).sum(axis=1)
    
    return ranks.sort_values('mean_rank')


# ============================================================================
# Ablation Analysis
# ============================================================================

def analyze_ablation(
    full_score: float,
    ablation_scores: Dict[str, float],
) -> pd.DataFrame:
    """
    Analyze ablation study results.
    
    Parameters
    ----------
    full_score : float
        Score of full model
    ablation_scores : Dict[str, float]
        Component -> score when removed
        
    Returns
    -------
    pd.DataFrame
        Analysis dataframe with impact metrics
    """
    results = []
    for component, score in ablation_scores.items():
        delta = score - full_score
        relative = delta / full_score * 100 if full_score != 0 else 0
        results.append({
            'component': component,
            'score': score,
            'delta': delta,
            'relative_delta': relative,
            'importance_rank': None,
        })
    
    df = pd.DataFrame(results)
    df['importance_rank'] = df['delta'].abs().rank(ascending=False).astype(int)
    df = df.sort_values('delta')
    
    return df


def compute_component_importance(
    ablation_results: Dict[str, float],
    full_score: float,
    method: str = 'shapley',
) -> Dict[str, float]:
    """
    Compute component importance scores.
    
    Parameters
    ----------
    ablation_results : Dict[str, float]
        Component -> score when removed
    full_score : float
        Score of full model
    method : str
        Importance method: 'delta', 'relative', 'shapley'
        
    Returns
    -------
    Dict[str, float]
        Component importance scores
    """
    if method == 'delta':
        return {k: full_score - v for k, v in ablation_results.items()}
    
    elif method == 'relative':
        return {k: (full_score - v) / full_score * 100 for k, v in ablation_results.items()}
    
    elif method == 'shapley':
        deltas = {k: full_score - v for k, v in ablation_results.items()}
        total_delta = sum(deltas.values())
        if total_delta == 0:
            return {k: 1.0 / len(deltas) for k in deltas}
        return {k: v / total_delta for k, v in deltas.items()}
    
    else:
        raise ValueError(f"Unknown importance method: {method}")


# ============================================================================
# Efficiency Analysis
# ============================================================================

def compute_pareto_frontier(
    methods: List[str],
    quality: List[float],
    efficiency: List[float],
    higher_quality_better: bool = True,
    higher_efficiency_better: bool = True,
) -> List[str]:
    """
    Identify methods on the Pareto frontier.
    
    Parameters
    ----------
    methods : List[str]
        Method names
    quality : List[float]
        Quality scores
    efficiency : List[float]
        Efficiency scores
    higher_quality_better : bool
        Whether higher quality is better
    higher_efficiency_better : bool
        Whether higher efficiency is better
        
    Returns
    -------
    List[str]
        Methods on the Pareto frontier
    """
    n = len(methods)
    pareto = []
    
    for i in range(n):
        dominated = False
        for j in range(n):
            if i == j:
                continue
            
            # Check if j dominates i
            q_better = (quality[j] > quality[i]) if higher_quality_better else (quality[j] < quality[i])
            e_better = (efficiency[j] > efficiency[i]) if higher_efficiency_better else (efficiency[j] < efficiency[i])
            q_equal = quality[j] == quality[i]
            e_equal = efficiency[j] == efficiency[i]
            
            if (q_better or q_equal) and (e_better or e_equal) and (q_better or e_better):
                dominated = True
                break
        
        if not dominated:
            pareto.append(methods[i])
    
    return pareto


def compute_efficiency_score(
    quality: float,
    params_percent: float,
    training_time: float,
    weights: Optional[Dict[str, float]] = None,
) -> float:
    """
    Compute composite efficiency score.
    
    Parameters
    ----------
    quality : float
        Quality metric (higher is better)
    params_percent : float
        Parameter percentage (lower is better)
    training_time : float
        Training time in hours (lower is better)
    weights : Dict[str, float], optional
        Weights for each component
        
    Returns
    -------
    float
        Composite efficiency score
    """
    if weights is None:
        weights = {'quality': 0.5, 'params': 0.25, 'time': 0.25}
    
    q_norm = quality / 100
    
    p_norm = 1 - min(params_percent / 5, 1)
    
    t_norm = 1 - min(training_time / 50, 1)
    
    return (weights['quality'] * q_norm + 
            weights['params'] * p_norm + 
            weights['time'] * t_norm)


# ============================================================================
# Data I/O
# ============================================================================

def load_results_json(path: Union[str, Path]) -> Dict:
    """Load results from JSON file."""
    with open(path, 'r') as f:
        return json.load(f)


def save_results_json(data: Dict, path: Union[str, Path]) -> None:
    """Save results to JSON file."""
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)


def load_results_csv(path: Union[str, Path]) -> pd.DataFrame:
    """Load results from CSV file."""
    return pd.read_csv(path)


def save_results_csv(df: pd.DataFrame, path: Union[str, Path]) -> None:
    """Save results to CSV file."""
    df.to_csv(path, index=False)


def results_to_latex_table(
    df: pd.DataFrame,
    caption: str = 'Results',
    label: str = 'tab:results',
    highlight_best: bool = True,
    precision: int = 1,
) -> str:
    """
    Convert results dataframe to LaTeX table.
    
    Parameters
    ----------
    df : pd.DataFrame
        Results dataframe
    caption : str
        Table caption
    label : str
        LaTeX label
    highlight_best : bool
        Whether to bold best values
    precision : int
        Decimal precision
        
    Returns
    -------
    str
        LaTeX table string
    """
    df_formatted = df.copy()
    
    # Format numeric columns
    for col in df_formatted.select_dtypes(include=[np.number]).columns:
        if highlight_best:
            best_idx = df_formatted[col].idxmax()
            df_formatted[col] = df_formatted[col].apply(lambda x: f'{x:.{precision}f}')
            df_formatted.loc[best_idx, col] = f"\\textbf{{{df_formatted.loc[best_idx, col]}}}"
        else:
            df_formatted[col] = df_formatted[col].apply(lambda x: f'{x:.{precision}f}')
    
    latex = df_formatted.to_latex(
        index=True,
        escape=False,
        caption=caption,
        label=label,
    )
    
    return latex
