"""
Reusable plotting functions for FLOWRA research paper figures.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import warnings

# ============================================================================
# Color Schemes
# ============================================================================

FLOWRA_COLORS = {
    'FLOWRA': '#8b5cf6',
    'DoRA': '#06b6d4',
    'LoRA': '#f97316',
    'AdaLoRA': '#22c55e',
    'FullFT': '#64748b',
    'Frozen': '#94a3b8',
    'BitFit': '#ec4899',
    'Adapter': '#14b8a6',
    'Prefix': '#f59e0b',
}

GRADIENT_PURPLE = ['#8b5cf6', '#a78bfa', '#c4b5fd', '#ddd6fe', '#ede9fe', '#f5f3ff']
GRADIENT_HEAT = ['#ef4444', '#f97316', '#eab308', '#22c55e', '#06b6d4', '#3b82f6']

# ============================================================================
# Style Configuration
# ============================================================================

def setup_style(font_size: int = 10, use_latex: bool = False):
    """Configure matplotlib style for publication-quality figures."""
    plt.style.use('seaborn-v0_8-whitegrid')
    
    config = {
        'font.family': 'DejaVu Sans',
        'font.size': font_size,
        'axes.titlesize': font_size + 2,
        'axes.labelsize': font_size,
        'xtick.labelsize': font_size - 1,
        'ytick.labelsize': font_size - 1,
        'legend.fontsize': font_size - 1,
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'savefig.facecolor': 'white',
        'savefig.dpi': 300,
        'axes.spines.top': False,
        'axes.spines.right': False,
    }
    
    if use_latex:
        config.update({
            'text.usetex': True,
            'font.family': 'serif',
        })
    
    plt.rcParams.update(config)


# ============================================================================
# Radar Chart Utilities
# ============================================================================

@dataclass
class RadarConfig:
    """Configuration for radar charts."""
    fill_alpha: float = 0.15
    line_width: float = 2.0
    marker_size: int = 6
    grid_color: str = '#e2e8f0'
    label_size: int = 10
    ylim: Tuple[float, float] = (0, 100)


def create_radar_chart(
    ax: plt.Axes,
    categories: List[str],
    data: Dict[str, List[float]],
    colors: Optional[Dict[str, str]] = None,
    config: Optional[RadarConfig] = None,
    title: Optional[str] = None,
) -> plt.Axes:
    """
    Create a radar/spider chart.
    
    Parameters
    ----------
    ax : plt.Axes
        Matplotlib axes with polar projection
    categories : List[str]
        Category labels for each axis
    data : Dict[str, List[float]]
        Method name -> values mapping
    colors : Dict[str, str], optional
        Method name -> color mapping
    config : RadarConfig, optional
        Chart configuration
    title : str, optional
        Chart title
        
    Returns
    -------
    plt.Axes
        The modified axes object
    """
    if colors is None:
        colors = FLOWRA_COLORS
    if config is None:
        config = RadarConfig()
    
    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, size=config.label_size)
    
    for method, values in data.items():
        vals = list(values) + [values[0]]
        color = colors.get(method, '#64748b')
        
        ax.plot(angles, vals, 'o-', 
                linewidth=config.line_width, 
                label=method, 
                color=color, 
                markersize=config.marker_size)
        ax.fill(angles, vals, alpha=config.fill_alpha, color=color)
    
    ax.set_ylim(*config.ylim)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    
    if title:
        ax.set_title(title, fontweight='bold', pad=20)
    
    return ax


def normalize_for_radar(
    data: Dict[str, List[float]],
    reference: Optional[List[float]] = None,
    invert: Optional[List[bool]] = None,
    scale: float = 100.0,
) -> Dict[str, List[float]]:
    """
    Normalize data for radar chart display.
    
    Parameters
    ----------
    data : Dict[str, List[float]]
        Raw data to normalize
    reference : List[float], optional
        Reference values (e.g., Full FT scores) for percentage calculation
    invert : List[bool], optional
        Whether to invert each metric (for metrics where lower is better)
    scale : float
        Maximum scale value (default 100)
        
    Returns
    -------
    Dict[str, List[float]]
        Normalized data
    """
    n_metrics = len(next(iter(data.values())))
    
    if reference is None:
        # Use max across all methods as reference
        reference = [max(data[m][i] for m in data) for i in range(n_metrics)]
    
    if invert is None:
        invert = [False] * n_metrics
    
    normalized = {}
    for method, values in data.items():
        norm_vals = []
        for i, (val, ref, inv) in enumerate(zip(values, reference, invert)):
            if inv:
                norm_val = (ref - val) / ref * scale if ref != 0 else 0
            else:
                norm_val = val / ref * scale if ref != 0 else 0
            norm_vals.append(min(norm_val, scale))  # Cap at scale
        normalized[method] = norm_vals
    
    return normalized


# ============================================================================
# Bar Chart Utilities
# ============================================================================

def grouped_bar_chart(
    ax: plt.Axes,
    categories: List[str],
    data: Dict[str, List[float]],
    colors: Optional[Dict[str, str]] = None,
    bar_width: float = 0.15,
    ylabel: str = 'Score',
    title: Optional[str] = None,
    show_values: bool = False,
    value_fmt: str = '{:.1f}',
) -> plt.Axes:
    """
    Create a grouped bar chart.
    
    Parameters
    ----------
    ax : plt.Axes
        Matplotlib axes
    categories : List[str]
        X-axis categories
    data : Dict[str, List[float]]
        Method name -> values mapping
    colors : Dict[str, str], optional
        Method name -> color mapping
    bar_width : float
        Width of individual bars
    ylabel : str
        Y-axis label
    title : str, optional
        Chart title
    show_values : bool
        Whether to show value labels on bars
    value_fmt : str
        Format string for value labels
        
    Returns
    -------
    plt.Axes
        The modified axes object
    """
    if colors is None:
        colors = FLOWRA_COLORS
    
    x = np.arange(len(categories))
    n_methods = len(data)
    
    for i, (method, values) in enumerate(data.items()):
        offset = bar_width * (i - n_methods / 2 + 0.5)
        color = colors.get(method, '#64748b')
        bars = ax.bar(x + offset, values, bar_width, 
                      label=method, color=color, edgecolor='white')
        
        if show_values:
            for bar, val in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                        value_fmt.format(val), ha='center', va='bottom', 
                        fontsize=8, rotation=0)
    
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_ylabel(ylabel)
    ax.legend(loc='upper right')
    
    if title:
        ax.set_title(title, fontweight='bold')
    
    return ax


def horizontal_bar_chart(
    ax: plt.Axes,
    categories: List[str],
    values: List[float],
    colors: Optional[List[str]] = None,
    xlabel: str = 'Value',
    title: Optional[str] = None,
    show_values: bool = True,
    value_fmt: str = '{:.1f}',
    value_position: str = 'inside',  # 'inside' or 'outside'
) -> plt.Axes:
    """
    Create a horizontal bar chart.
    
    Parameters
    ----------
    ax : plt.Axes
        Matplotlib axes
    categories : List[str]
        Y-axis categories
    values : List[float]
        Bar values
    colors : List[str], optional
        Bar colors
    xlabel : str
        X-axis label
    title : str, optional
        Chart title
    show_values : bool
        Whether to show value labels
    value_fmt : str
        Format string for value labels
    value_position : str
        Position of value labels ('inside' or 'outside')
        
    Returns
    -------
    plt.Axes
        The modified axes object
    """
    if colors is None:
        colors = [FLOWRA_COLORS.get(cat, '#64748b') for cat in categories]
    
    bars = ax.barh(categories, values, color=colors, edgecolor='white', height=0.6)
    ax.set_xlabel(xlabel)
    
    if show_values:
        for bar, val in zip(bars, values):
            if value_position == 'inside':
                x_pos = val - 0.3 if val > 0 else val + 0.1
                ha = 'right' if val > 0 else 'left'
                color = 'white'
            else:
                x_pos = val + 0.1 if val > 0 else val - 0.1
                ha = 'left' if val > 0 else 'right'
                color = 'black'
            
            ax.text(x_pos, bar.get_y() + bar.get_height()/2,
                    value_fmt.format(val), va='center', ha=ha,
                    fontsize=9, fontweight='bold', color=color)
    
    if title:
        ax.set_title(title, fontweight='bold')
    
    return ax


# ============================================================================
# Line Chart Utilities
# ============================================================================

def convergence_plot(
    ax: plt.Axes,
    x_values: List[Union[int, float]],
    data: Dict[str, List[float]],
    colors: Optional[Dict[str, str]] = None,
    xlabel: str = 'Training Steps',
    ylabel: str = 'Accuracy %',
    title: Optional[str] = None,
    fill_between: Optional[Tuple[str, str]] = None,
    markers: bool = True,
) -> plt.Axes:
    """
    Create a convergence/training curve plot.
    
    Parameters
    ----------
    ax : plt.Axes
        Matplotlib axes
    x_values : List[Union[int, float]]
        X-axis values (e.g., training steps)
    data : Dict[str, List[float]]
        Method name -> values mapping
    colors : Dict[str, str], optional
        Method name -> color mapping
    xlabel : str
        X-axis label
    ylabel : str
        Y-axis label
    title : str, optional
        Chart title
    fill_between : Tuple[str, str], optional
        Two method names to fill between
    markers : bool
        Whether to show markers
        
    Returns
    -------
    plt.Axes
        The modified axes object
    """
    if colors is None:
        colors = FLOWRA_COLORS
    
    marker = 'o' if markers else None
    
    for method, values in data.items():
        color = colors.get(method, '#64748b')
        ax.plot(x_values, values, marker=marker, label=method,
                linewidth=2.5, markersize=6, color=color)
    
    if fill_between:
        method1, method2 = fill_between
        if method1 in data and method2 in data:
            color = colors.get(method1, '#64748b')
            ax.fill_between(x_values, data[method1], data[method2],
                           alpha=0.1, color=color)
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(loc='lower right')
    
    if title:
        ax.set_title(title, fontweight='bold')
    
    return ax


# ============================================================================
# Dual-Axis Plot Utilities
# ============================================================================

def dual_axis_plot(
    ax: plt.Axes,
    x_values: List,
    y1_data: Dict[str, List[float]],
    y2_data: Dict[str, List[float]],
    y1_label: str = 'Primary',
    y2_label: str = 'Secondary',
    y1_color: str = '#8b5cf6',
    y2_color: str = '#ef4444',
    title: Optional[str] = None,
) -> Tuple[plt.Axes, plt.Axes]:
    """
    Create a dual-axis plot.
    
    Parameters
    ----------
    ax : plt.Axes
        Primary matplotlib axes
    x_values : List
        X-axis values
    y1_data : Dict[str, List[float]]
        Data for primary y-axis
    y2_data : Dict[str, List[float]]
        Data for secondary y-axis
    y1_label : str
        Primary y-axis label
    y2_label : str
        Secondary y-axis label
    y1_color : str
        Primary axis color
    y2_color : str
        Secondary axis color
    title : str, optional
        Chart title
        
    Returns
    -------
    Tuple[plt.Axes, plt.Axes]
        Primary and secondary axes
    """
    ax2 = ax.twinx()
    
    for name, values in y1_data.items():
        ax.plot(x_values, values, 'o-', color=y1_color, linewidth=2, 
                markersize=6, label=name)
    
    for name, values in y2_data.items():
        ax2.plot(x_values, values, 's--', color=y2_color, linewidth=2,
                markersize=6, label=name)
    
    ax.set_ylabel(y1_label, color=y1_color)
    ax2.set_ylabel(y2_label, color=y2_color)
    ax.tick_params(axis='y', labelcolor=y1_color)
    ax2.tick_params(axis='y', labelcolor=y2_color)
    
    # Combine legends
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    
    if title:
        ax.set_title(title, fontweight='bold')
    
    return ax, ax2


# ============================================================================
# Annotation Utilities
# ============================================================================

def add_annotation(
    ax: plt.Axes,
    text: str,
    xy: Tuple[float, float],
    xytext: Tuple[float, float],
    color: str = '#22c55e',
    fontsize: int = 10,
    fontweight: str = 'bold',
    arrow: bool = True,
) -> None:
    """Add an annotation with optional arrow."""
    arrowprops = dict(arrowstyle='->', color=color) if arrow else None
    ax.annotate(text, xy=xy, xytext=xytext,
                fontsize=fontsize, color=color, fontweight=fontweight,
                arrowprops=arrowprops)


def add_reference_line(
    ax: plt.Axes,
    value: float,
    orientation: str = 'horizontal',
    color: str = '#64748b',
    linestyle: str = '--',
    alpha: float = 0.5,
    label: Optional[str] = None,
) -> None:
    """Add a reference line to the plot."""
    if orientation == 'horizontal':
        ax.axhline(y=value, color=color, linestyle=linestyle, alpha=alpha, label=label)
    else:
        ax.axvline(x=value, color=color, linestyle=linestyle, alpha=alpha, label=label)


# ============================================================================
# Figure Export Utilities  
# ============================================================================

def save_figure(
    fig: plt.Figure,
    filename: str,
    formats: List[str] = ['png', 'pdf'],
    dpi: int = 300,
    tight: bool = True,
) -> List[str]:
    """
    Save figure in multiple formats.
    
    Parameters
    ----------
    fig : plt.Figure
        Figure to save
    filename : str
        Base filename (without extension)
    formats : List[str]
        Output formats
    dpi : int
        Resolution for raster formats
    tight : bool
        Whether to use tight bounding box
        
    Returns
    -------
    List[str]
        List of saved file paths
    """
    saved = []
    bbox = 'tight' if tight else None
    
    for fmt in formats:
        path = f'{filename}.{fmt}'
        fig.savefig(path, format=fmt, dpi=dpi, bbox_inches=bbox)
        saved.append(path)
    
    return saved


# ============================================================================
# Quick Plot Functions
# ============================================================================

def quick_comparison_radar(
    data: Dict[str, Dict[str, float]],
    title: str = 'Method Comparison',
    figsize: Tuple[int, int] = (8, 8),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Quick function to create a comparison radar chart.
    
    Parameters
    ----------
    data : Dict[str, Dict[str, float]]
        Nested dict: method -> {metric: value}
    title : str
        Chart title
    figsize : Tuple[int, int]
        Figure size
    save_path : str, optional
        Path to save figure
        
    Returns
    -------
    plt.Figure
        The created figure
    """
    setup_style()
    
    # Extract categories and reorganize data
    categories = list(next(iter(data.values())).keys())
    plot_data = {method: [metrics[cat] for cat in categories] 
                 for method, metrics in data.items()}
    
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='polar')
    
    create_radar_chart(ax, categories, plot_data, title=title)
    
    plt.tight_layout()
    
    if save_path:
        save_figure(fig, save_path)
    
    return fig


def quick_ablation_bar(
    variants: List[str],
    scores: List[float],
    baseline_idx: int = 0,
    title: str = 'Ablation Study',
    figsize: Tuple[int, int] = (10, 5),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Quick function to create an ablation study bar chart.
    
    Parameters
    ----------
    variants : List[str]
        Variant names
    scores : List[float]
        Performance scores
    baseline_idx : int
        Index of baseline variant
    title : str
        Chart title
    figsize : Tuple[int, int]
        Figure size
    save_path : str, optional
        Path to save figure
        
    Returns
    -------
    plt.Figure
        The created figure
    """
    setup_style()
    
    fig, ax = plt.subplots(figsize=figsize)
    
    baseline_score = scores[baseline_idx]
    deltas = [s - baseline_score for s in scores]
    
    # Color gradient based on performance
    colors = [FLOWRA_COLORS['FLOWRA'] if i == baseline_idx 
              else GRADIENT_PURPLE[min(i, len(GRADIENT_PURPLE)-1)]
              for i in range(len(variants))]
    
    bars = ax.bar(variants, scores, color=colors, edgecolor='white')
    
    # Add delta labels
    for bar, delta, score in zip(bars, deltas, scores):
        color = '#22c55e' if delta >= 0 else '#ef4444'
        label = f'{delta:+.1f}' if delta != 0 else 'Base'
        ax.text(bar.get_x() + bar.get_width()/2, score + 0.2,
                label, ha='center', va='bottom', fontsize=9,
                color=color, fontweight='bold')
    
    ax.set_ylabel('Score')
    ax.set_title(title, fontweight='bold')
    
    # Add baseline reference line
    add_reference_line(ax, baseline_score, color=FLOWRA_COLORS['FLOWRA'])
    
    plt.tight_layout()
    
    if save_path:
        save_figure(fig, save_path)
    
    return fig
