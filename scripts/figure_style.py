"""
Publication-Grade Typography and Colorblind-Safe Palette for CoRL Figures.

Strict adherence to repository guidelines:
- Zero AST fallbacks.
- Serif typography matching Computer Modern / publication standards.
- Consistent colorblind-safe categorical palette.
- Multi-format vector (.pdf, .svg) and 300+ DPI lossless raster (.png) export.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Sequence, Union
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ==============================================================================
# Colorblind-Safe Categorical Palette
# ==============================================================================
PALETTE: Dict[str, str] = {
    # Proposed Model
    "spwm": "#1E3A8A",            # Deep Navy Sapphire (Primary proposed model)
    "spwm_light": "#3B82F6",      # Bright Blue (Rollouts / Accents)
    "spwm_basin": "#DBEAFE",      # Soft Ice Blue (Attractor manifold / Bounds)
    
    # Baselines & Ablations
    "snn_matched": "#0D9488",     # Deep Teal (SNN-ALIF matched baseline)
    "ann_rate": "#DC2626",        # Crimson Red (Parameter-matched continuous ablation)
    "ann_rate_light": "#FCA5A5",  # Coral Pink (Diverging rollout envelope)
    "continuous_rssm": "#EA580C", # Deep Orange (Continuous RSSM)
    "deep_gru": "#7C3AED",        # Purple / Violet (Deep GRU)
    "conv_lstm": "#475569",       # Charcoal Slate (Continuous ConvLSTM)
    
    # Ground Truth & Reference
    "ground_truth": "#0F172A",    # Dark Charcoal / Obsidian
    "grid": "#E2E8F0",            # Subtle Slate Grid
    "heft_region": "#FEF3C7",     # Soft Amber (Active perception / heft window)
    "heft_border": "#F59E0B",     # Amber border
    
    # Hardware / Neuromorphic
    "gpu": "#DC2626",             # Nvidia Orin GPU
    "neuromorphic": "#0D9488",    # Intel Loihi-2 Neuromorphic
}


def set_paper_style() -> None:
    """Configures matplotlib rcParams for publication-grade serif figures."""
    plt.rcParams.update({
        # Typography: Serif with Computer Modern math font
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Times New Roman", "Computer Modern Roman"],
        "mathtext.fontset": "cm",
        "axes.unicode_minus": False,
        
        # Font Sizes (matching CoRL / IEEE double-column format)
        "font.size": 9.5,
        "axes.titlesize": 10.5,
        "axes.labelsize": 9.5,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "legend.fontsize": 8.0,
        "figure.titlesize": 11.5,
        
        # Line widths and styling
        "lines.linewidth": 1.75,
        "lines.markersize": 5.0,
        "patch.linewidth": 1.0,
        
        # Grid & Axes Styling
        "axes.linewidth": 0.8,
        "axes.edgecolor": "#334155",
        "axes.grid": True,
        "grid.color": "#CBD5E1",
        "grid.linestyle": ":",
        "grid.linewidth": 0.5,
        "grid.alpha": 0.7,
        
        # Export defaults
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.04,
    })


def save_publication_figure(
    fig: plt.Figure,
    base_path: Union[str, Path],
    formats: Sequence[str] = ("pdf", "svg", "png"),
    dpi: int = 300,
) -> None:
    """
    Exports figure to multiple formats (.pdf, .svg, .png) with 300+ DPI.
    
    Args:
        fig: Matplotlib Figure instance.
        base_path: Path without extension or with any extension (will be stripped).
        formats: Target file formats to generate.
        dpi: Dots per inch for raster export.
    """
    p = Path(base_path)
    stem = p.parent / p.stem
    stem.parent.mkdir(parents=True, exist_ok=True)
    
    for fmt in formats:
        target = stem.with_suffix(f".{fmt}")
        fig.savefig(target, format=fmt, dpi=dpi, bbox_inches="tight")
        print(f"[Figure Export] Saved: {target.resolve()}")


__all__ = ["PALETTE", "set_paper_style", "save_publication_figure"]

