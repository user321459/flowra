"""
Functions for benchmark data handling and results aggregation.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from pathlib import Path
import json


# ============================================================================
# Benchmark Metadata
# ============================================================================

GLUE_TASKS = {
    'mnli': {'metric': 'accuracy', 'num_labels': 3, 'task_type': 'nli'},
    'qqp': {'metric': 'f1', 'num_labels': 2, 'task_type': 'paraphrase'},
    'qnli': {'metric': 'accuracy', 'num_labels': 2, 'task_type': 'nli'},
    'sst2': {'metric': 'accuracy', 'num_labels': 2, 'task_type': 'sentiment'},
    'cola': {'metric': 'matthews_corr', 'num_labels': 2, 'task_type': 'acceptability'},
    'stsb': {'metric': 'spearman_corr', 'num_labels': 1, 'task_type': 'similarity'},
    'mrpc': {'metric': 'f1', 'num_labels': 2, 'task_type': 'paraphrase'},
    'rte': {'metric': 'accuracy', 'num_labels': 2, 'task_type': 'nli'},
}

SUPERGLUE_TASKS = {
    'boolq': {'metric': 'accuracy', 'task_type': 'qa'},
    'cb': {'metric': 'f1', 'task_type': 'nli'},
    'copa': {'metric': 'accuracy', 'task_type': 'reasoning'},
    'multirc': {'metric': 'f1a', 'task_type': 'qa'},
    'record': {'metric': 'f1', 'task_type': 'qa'},
    'rte': {'metric': 'accuracy', 'task_type': 'nli'},
    'wic': {'metric': 'accuracy', 'task_type': 'disambiguation'},
    'wsc': {'metric': 'accuracy', 'task_type': 'coreference'},
}

CODE_BENCHMARKS = {
    'humaneval': {'metric': 'pass@1', 'language': 'python'},
    'mbpp': {'metric': 'pass@1', 'language': 'python'},
}

MATH_BENCHMARKS = {
    'gsm8k': {'metric': 'accuracy', 'shots': 8},
    'math': {'metric': 'accuracy', 'shots': 4},
}

VISION_BENCHMARKS = {
    'imagenet': {'metric': 'top1_accuracy', 'num_classes': 1000},
    'cub200': {'metric': 'accuracy', 'num_classes': 200},
    'ade20k': {'metric': 'miou', 'task': 'segmentation'},
}


# ============================================================================
# Results Data Structures
# ============================================================================

@dataclass
class TaskResult:
    """Result for a single task."""
    task: str
    score: float
    metric: str
    std: Optional[float] = None
    num_samples: Optional[int] = None
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BenchmarkResult:
    """Aggregated benchmark result."""
    benchmark: str
    method: str
    model: str
    task_results: List[TaskResult] = field(default_factory=list)
    config: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def average_score(self) -> float:
        if not self.task_results:
            return 0.0
        return np.mean([r.score for r in self.task_results])
    
    @property
    def task_scores(self) -> Dict[str, float]:
        return {r.task: r.score for r in self.task_results}
    
    def to_dict(self) -> Dict:
        return {
            'benchmark': self.benchmark,
            'method': self.method,
            'model': self.model,
            'average': self.average_score,
            'tasks': self.task_scores,
            'config': self.config,
        }

# ============================================================================
# Results Aggregation
# ============================================================================

def compute_benchmark_average(
    task_scores: Dict[str, float],
    weights: Optional[Dict[str, float]] = None,
) -> float:
    """Compute weighted average across tasks."""
    if weights is None:
        return np.mean(list(task_scores.values()))
    
    total_weight = sum(weights.get(t, 1.0) for t in task_scores)
    weighted_sum = sum(s * weights.get(t, 1.0) for t, s in task_scores.items())
    return weighted_sum / total_weight


def aggregate_results(
    results: List[BenchmarkResult],
    group_by: str = 'method',
) -> pd.DataFrame:
    """Aggregate multiple benchmark results."""
    rows = []
    for r in results:
        row = {
            'method': r.method,
            'model': r.model,
            'benchmark': r.benchmark,
            'average': r.average_score,
        }
        row.update(r.task_scores)
        rows.append(row)
    
    df = pd.DataFrame(rows)
    
    if group_by:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df = df.groupby(group_by)[numeric_cols].mean().reset_index()
    
    return df


def compare_methods(
    reference: str = 'Full FT',
    results: Optional[Dict[str, Dict[str, float]]] = None,
) -> pd.DataFrame:
    """Compare methods against reference."""
    if results is None:
        results = GLUE_REFERENCE_RESULTS
    
    ref_scores = results[reference]
    
    rows = []
    for method, scores in results.items():
        row = {'method': method}
        for task, score in scores.items():
            row[task] = score
            row[f'{task}_delta'] = score - ref_scores[task]
        row['average'] = np.mean(list(scores.values()))
        row['avg_delta'] = row['average'] - np.mean(list(ref_scores.values()))
        rows.append(row)
    
    return pd.DataFrame(rows)


# ============================================================================
# Results I/O
# ============================================================================

def save_results_to_json(results: List[BenchmarkResult], path: Path) -> None:
    """Save benchmark results to JSON file."""
    data = [r.to_dict() for r in results]
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)


def load_results_from_json(path: Path) -> List[BenchmarkResult]:
    """Load benchmark results from JSON file."""
    with open(path, 'r') as f:
        data = json.load(f)
    
    results = []
    for d in data:
        task_results = [
            TaskResult(task=t, score=s, metric='unknown')
            for t, s in d.get('tasks', {}).items()
        ]
        results.append(BenchmarkResult(
            benchmark=d['benchmark'],
            method=d['method'],
            model=d['model'],
            task_results=task_results,
            config=d.get('config', {}),
        ))
    
    return results


# ============================================================================
# Results Formatting
# ============================================================================

def format_results_table(
    results: Dict[str, Dict[str, float]],
    methods_order: Optional[List[str]] = None,
    tasks_order: Optional[List[str]] = None,
    highlight_best: bool = True,
    precision: int = 1,
) -> str:
    """Format results as a text table."""
    if methods_order is None:
        methods_order = list(results.keys())
    if tasks_order is None:
        tasks_order = list(next(iter(results.values())).keys())
    
    # Find best per task
    best_per_task = {}
    if highlight_best:
        for task in tasks_order:
            scores = [results[m].get(task, 0) for m in methods_order]
            best_per_task[task] = max(scores)
    
    # Build table
    col_width = max(len(t) for t in tasks_order + ['Method', 'Avg']) + 2
    
    # Header
    header = 'Method'.ljust(col_width)
    for task in tasks_order:
        header += task.ljust(col_width)
    header += 'Avg'.ljust(col_width)
    
    lines = [header, '-' * len(header)]
    
    # Rows
    for method in methods_order:
        row = method.ljust(col_width)
        scores = []
        for task in tasks_order:
            score = results[method].get(task, 0)
            scores.append(score)
            formatted = f'{score:.{precision}f}'
            if highlight_best and score == best_per_task[task]:
                formatted = f'**{formatted}**'
            row += formatted.ljust(col_width)
        
        avg = np.mean(scores)
        row += f'{avg:.{precision}f}'.ljust(col_width)
        lines.append(row)
    
    return '\n'.join(lines)


def results_to_markdown(
    results: Dict[str, Dict[str, float]],
    caption: str = 'Results',
) -> str:
    """Convert results to Markdown table."""
    methods = list(results.keys())
    tasks = list(next(iter(results.values())).keys())
    
    header = '| Method | ' + ' | '.join(tasks) + ' | Avg |'
    separator = '|' + '|'.join(['---'] * (len(tasks) + 2)) + '|'
    
    rows = []
    for method in methods:
        scores = [results[method].get(t, 0) for t in tasks]
        avg = np.mean(scores)
        row = f'| {method} | ' + ' | '.join(f'{s:.1f}' for s in scores) + f' | {avg:.1f} |'
        rows.append(row)
    
    return f'**{caption}**\n\n{header}\n{separator}\n' + '\n'.join(rows)


def results_to_latex(
    results: Dict[str, Dict[str, float]],
    caption: str = 'Results',
    label: str = 'tab:results',
    highlight_best: bool = True,
) -> str:
    """Convert results to LaTeX table."""
    methods = list(results.keys())
    tasks = list(next(iter(results.values())).keys())
    
    # Find best per task
    best_per_task = {}
    for task in tasks:
        scores = [results[m].get(task, 0) for m in methods]
        best_per_task[task] = max(scores)
    
    # Build LaTeX
    cols = 'l' + 'c' * (len(tasks) + 1)
    header = 'Method & ' + ' & '.join(tasks) + ' & Avg \\\\'
    
    rows = []
    for method in methods:
        scores = [results[method].get(t, 0) for t in tasks]
        avg = np.mean(scores)
        
        formatted = []
        for task, score in zip(tasks, scores):
            s = f'{score:.1f}'
            if highlight_best and score == best_per_task[task]:
                s = f'\\textbf{{{s}}}'
            formatted.append(s)
        
        row = f'{method} & ' + ' & '.join(formatted) + f' & {avg:.1f} \\\\'
        rows.append(row)
    
    latex = f"""\\begin{{table}}[h]
\\centering
\\caption{{{caption}}}
\\label{{{label}}}
\\begin{{tabular}}{{{cols}}}
\\toprule
{header}
\\midrule
{chr(10).join(rows)}
\\bottomrule
\\end{{tabular}}
\\end{{table}}"""
    
    return latex


# ============================================================================
# Statistical Comparison
# ============================================================================

def paired_comparison(
    method1_scores: List[float],
    method2_scores: List[float],
    test: str = 'wilcoxon',
) -> Dict[str, float]:
    """Perform paired statistical comparison between two methods."""
    try:
        from scipy import stats
    except ImportError:
        return {'error': 'scipy not installed'}
    
    scores1 = np.array(method1_scores)
    scores2 = np.array(method2_scores)
    diff = scores1 - scores2
    
    results = {
        'mean_diff': np.mean(diff),
        'std_diff': np.std(diff),
        'method1_mean': np.mean(scores1),
        'method2_mean': np.mean(scores2),
    }
    
    if test == 'wilcoxon':
        stat, p_value = stats.wilcoxon(scores1, scores2)
        results['statistic'] = stat
        results['p_value'] = p_value
        results['test'] = 'Wilcoxon signed-rank'
    elif test == 'ttest':
        stat, p_value = stats.ttest_rel(scores1, scores2)
        results['statistic'] = stat
        results['p_value'] = p_value
        results['test'] = 'Paired t-test'
    
    results['significant'] = results.get('p_value', 1.0) < 0.05
    
    return results


def bootstrap_confidence_interval(
    scores: List[float],
    confidence: float = 0.95,
    n_bootstrap: int = 1000,
) -> Tuple[float, float]:
    """Compute bootstrap confidence interval."""
    scores = np.array(scores)
    n = len(scores)
    
    bootstrap_means = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(scores, size=n, replace=True)
        bootstrap_means.append(np.mean(sample))
    
    alpha = 1 - confidence
    lower = np.percentile(bootstrap_means, alpha/2 * 100)
    upper = np.percentile(bootstrap_means, (1 - alpha/2) * 100)
    
    return lower, upper
