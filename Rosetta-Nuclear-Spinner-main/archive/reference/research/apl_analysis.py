#!/usr/bin/env python3
"""
APL Test Results Analyzer — Phase 4 Enhancement
================================================

Automated statistical analysis and visualization for APL Seven Sentences
test results. Provides publication-ready figures and comprehensive
statistical summaries.

Features:
- Statistical significance testing (t-test, Mann-Whitney U)
- Effect size calculation (Cohen's d, rank-biserial)
- Confidence intervals (bootstrap)
- Multiple comparison correction (Bonferroni, FDR)
- Visualization: histograms, bar charts, forest plots
- Export: LaTeX tables, CSV, JSON

Usage:
  python apl_analysis.py results/apl_test_results.json --plot --export
  python apl_analysis.py results/*.json --compare --output report/

@version 2.0.0 (TRIAD Protocol Phase 4)
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import warnings

# Optional imports
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    np = None

try:
    from scipy import stats as scipy_stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    plt = None


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class ComparisonResult:
    """Statistical comparison between LHS and Control conditions."""
    sentence_id: str
    n_lhs: int
    n_control: int
    
    # Descriptive statistics
    lhs_mean: float
    lhs_std: float
    lhs_median: Optional[float] = None
    control_mean: float = 0.0
    control_std: float = 0.0
    control_median: Optional[float] = None
    
    # Effect sizes
    cohens_d: Optional[float] = None
    rank_biserial: Optional[float] = None
    
    # Parametric tests
    t_statistic: Optional[float] = None
    t_pvalue: Optional[float] = None
    
    # Non-parametric tests
    u_statistic: Optional[float] = None
    u_pvalue: Optional[float] = None
    
    # Confidence intervals (95%)
    mean_diff: Optional[float] = None
    ci_lower: Optional[float] = None
    ci_upper: Optional[float] = None
    
    # Corrected p-values
    bonferroni_p: Optional[float] = None
    fdr_p: Optional[float] = None
    
    def is_significant(self, alpha: float = 0.05, use_corrected: bool = True) -> bool:
        """Check significance using appropriate p-value."""
        if use_corrected and self.bonferroni_p is not None:
            return self.bonferroni_p < alpha
        return self.t_pvalue is not None and self.t_pvalue < alpha
    
    def effect_magnitude(self) -> str:
        """Interpret effect size magnitude."""
        if self.cohens_d is None:
            return "unknown"
        d = abs(self.cohens_d)
        if d < 0.2:
            return "negligible"
        elif d < 0.5:
            return "small"
        elif d < 0.8:
            return "medium"
        else:
            return "large"


@dataclass
class AnalysisReport:
    """Complete analysis report for a test run."""
    source_file: str
    timestamp: str
    comparisons: List[ComparisonResult] = field(default_factory=list)
    summary: Dict[str, Any] = field(default_factory=dict)


# ============================================================================
# STATISTICAL FUNCTIONS
# ============================================================================

def cohens_d(group1: List[float], group2: List[float]) -> float:
    """
    Calculate Cohen's d effect size.
    
    Uses pooled standard deviation formula.
    """
    if not HAS_NUMPY:
        # Pure Python implementation
        n1, n2 = len(group1), len(group2)
        mean1 = sum(group1) / n1
        mean2 = sum(group2) / n2
        var1 = sum((x - mean1)**2 for x in group1) / (n1 - 1)
        var2 = sum((x - mean2)**2 for x in group2) / (n2 - 1)
        pooled_std = math.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
        if pooled_std == 0:
            return 0.0
        return (mean1 - mean2) / pooled_std
    
    arr1, arr2 = np.array(group1), np.array(group2)
    n1, n2 = len(arr1), len(arr2)
    var1, var2 = arr1.var(ddof=1), arr2.var(ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled_std == 0:
        return 0.0
    return float((arr1.mean() - arr2.mean()) / pooled_std)


def bootstrap_ci(
    group1: List[float],
    group2: List[float],
    n_bootstrap: int = 10000,
    alpha: float = 0.05,
    seed: Optional[int] = None
) -> Tuple[float, float, float]:
    """
    Bootstrap confidence interval for mean difference.
    
    Returns: (mean_diff, ci_lower, ci_upper)
    """
    if not HAS_NUMPY:
        mean_diff = sum(group1) / len(group1) - sum(group2) / len(group2)
        return mean_diff, mean_diff, mean_diff  # Can't bootstrap without numpy
    
    rng = np.random.default_rng(seed)
    arr1, arr2 = np.array(group1), np.array(group2)
    
    diffs = []
    for _ in range(n_bootstrap):
        sample1 = rng.choice(arr1, size=len(arr1), replace=True)
        sample2 = rng.choice(arr2, size=len(arr2), replace=True)
        diffs.append(sample1.mean() - sample2.mean())
    
    diffs = np.array(diffs)
    mean_diff = float(arr1.mean() - arr2.mean())
    ci_lower = float(np.percentile(diffs, 100 * alpha / 2))
    ci_upper = float(np.percentile(diffs, 100 * (1 - alpha / 2)))
    
    return mean_diff, ci_lower, ci_upper


def bonferroni_correction(p_values: List[float], alpha: float = 0.05) -> List[float]:
    """Apply Bonferroni correction to p-values."""
    n = len(p_values)
    return [min(p * n, 1.0) for p in p_values]


def fdr_correction(p_values: List[float], alpha: float = 0.05) -> List[float]:
    """Apply Benjamini-Hochberg FDR correction."""
    n = len(p_values)
    if n == 0:
        return []
    
    # Sort p-values with original indices
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    
    # Compute FDR-adjusted p-values
    adjusted = [0.0] * n
    cummin = float('inf')
    
    for rank, (orig_idx, p) in enumerate(reversed(indexed), 1):
        adj_p = p * n / (n - rank + 1)
        cummin = min(cummin, adj_p)
        adjusted[orig_idx] = min(cummin, 1.0)
    
    return adjusted


# ============================================================================
# ANALYZER CLASS
# ============================================================================

class APLResultsAnalyzer:
    """
    Comprehensive statistical analyzer for APL test results.
    """
    
    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha
        self.reports: List[AnalysisReport] = []
    
    def load_results(self, filepath: Path) -> AnalysisReport:
        """Load and analyze results from JSON file."""
        with open(filepath) as f:
            data = json.load(f)
        
        report = AnalysisReport(
            source_file=str(filepath),
            timestamp=data.get('timestamp', datetime.now().isoformat())
        )
        
        sentences = data.get('sentences', {})
        
        for sid, sdata in sentences.items():
            # Extract raw metrics if available, otherwise use summary stats
            lhs_metrics = sdata.get('lhs_metrics', [])
            ctrl_metrics = sdata.get('control_metrics', [])
            
            # If raw metrics not available, we can't do detailed analysis
            # but can still report the summary stats
            comparison = ComparisonResult(
                sentence_id=sid,
                n_lhs=sdata.get('n_lhs_trials', len(lhs_metrics)),
                n_control=sdata.get('n_control_trials', len(ctrl_metrics)),
                lhs_mean=sdata.get('lhs_mean', 0),
                lhs_std=sdata.get('lhs_std', 0),
                control_mean=sdata.get('control_mean', 0),
                control_std=sdata.get('control_std', 0),
                t_statistic=sdata.get('t_statistic'),
                t_pvalue=sdata.get('p_value'),
                cohens_d=sdata.get('effect_size')
            )
            
            # Compute additional statistics if raw data available
            if lhs_metrics and ctrl_metrics:
                self._compute_detailed_stats(comparison, lhs_metrics, ctrl_metrics)
            
            report.comparisons.append(comparison)
        
        # Apply multiple comparison corrections
        self._apply_corrections(report)
        
        # Generate summary
        report.summary = self._generate_summary(report)
        
        self.reports.append(report)
        return report
    
    def _compute_detailed_stats(
        self,
        comparison: ComparisonResult,
        lhs: List[float],
        ctrl: List[float]
    ) -> None:
        """Compute detailed statistics from raw metrics."""
        # Medians
        if HAS_NUMPY:
            comparison.lhs_median = float(np.median(lhs))
            comparison.control_median = float(np.median(ctrl))
        
        # Effect size
        comparison.cohens_d = cohens_d(lhs, ctrl)
        
        # Parametric t-test
        if HAS_SCIPY:
            t_stat, t_pval = scipy_stats.ttest_ind(lhs, ctrl, equal_var=False)
            comparison.t_statistic = float(t_stat)
            comparison.t_pvalue = float(t_pval)
            
            # Non-parametric Mann-Whitney U
            u_stat, u_pval = scipy_stats.mannwhitneyu(lhs, ctrl, alternative='two-sided')
            comparison.u_statistic = float(u_stat)
            comparison.u_pvalue = float(u_pval)
            
            # Rank-biserial correlation (effect size for U test)
            n1, n2 = len(lhs), len(ctrl)
            comparison.rank_biserial = float(1 - (2 * u_stat) / (n1 * n2))
        
        # Bootstrap CI
        mean_diff, ci_lo, ci_hi = bootstrap_ci(lhs, ctrl, seed=42)
        comparison.mean_diff = mean_diff
        comparison.ci_lower = ci_lo
        comparison.ci_upper = ci_hi
    
    def _apply_corrections(self, report: AnalysisReport) -> None:
        """Apply multiple comparison corrections."""
        p_values = [c.t_pvalue for c in report.comparisons if c.t_pvalue is not None]
        
        if not p_values:
            return
        
        bonf = bonferroni_correction(p_values, self.alpha)
        fdr = fdr_correction(p_values, self.alpha)
        
        idx = 0
        for comp in report.comparisons:
            if comp.t_pvalue is not None:
                comp.bonferroni_p = bonf[idx]
                comp.fdr_p = fdr[idx]
                idx += 1
    
    def _generate_summary(self, report: AnalysisReport) -> Dict[str, Any]:
        """Generate summary statistics across all comparisons."""
        comparisons = report.comparisons
        
        significant_uncorrected = sum(1 for c in comparisons if c.t_pvalue and c.t_pvalue < self.alpha)
        significant_bonferroni = sum(1 for c in comparisons if c.bonferroni_p and c.bonferroni_p < self.alpha)
        significant_fdr = sum(1 for c in comparisons if c.fdr_p and c.fdr_p < self.alpha)
        
        effect_sizes = [c.cohens_d for c in comparisons if c.cohens_d is not None]
        
        return {
            'n_comparisons': len(comparisons),
            'significant_uncorrected': significant_uncorrected,
            'significant_bonferroni': significant_bonferroni,
            'significant_fdr': significant_fdr,
            'mean_effect_size': sum(effect_sizes) / len(effect_sizes) if effect_sizes else None,
            'alpha': self.alpha
        }
    
    def print_report(self, report: AnalysisReport) -> None:
        """Print formatted analysis report."""
        print(f"\n{'='*70}")
        print(f"APL TEST ANALYSIS REPORT")
        print(f"{'='*70}")
        print(f"Source: {report.source_file}")
        print(f"Generated: {datetime.now().isoformat()}")
        print()
        
        # Individual comparisons
        print(f"{'Sentence':<10} {'LHS Mean':>10} {'Ctrl Mean':>10} {'Cohen d':>10} {'p-value':>10} {'Sig':>5}")
        print('-' * 70)
        
        for comp in report.comparisons:
            sig_marker = ''
            if comp.t_pvalue:
                if comp.t_pvalue < 0.001:
                    sig_marker = '***'
                elif comp.t_pvalue < 0.01:
                    sig_marker = '**'
                elif comp.t_pvalue < 0.05:
                    sig_marker = '*'
            
            d_str = f"{comp.cohens_d:.3f}" if comp.cohens_d else "N/A"
            p_str = f"{comp.t_pvalue:.4f}" if comp.t_pvalue else "N/A"
            
            print(f"{comp.sentence_id:<10} {comp.lhs_mean:>10.4f} {comp.control_mean:>10.4f} "
                  f"{d_str:>10} {p_str:>10} {sig_marker:>5}")
        
        # Summary
        print()
        print('-' * 70)
        print("SUMMARY")
        print('-' * 70)
        s = report.summary
        print(f"Total comparisons: {s['n_comparisons']}")
        print(f"Significant (p < {s['alpha']:.2f}):")
        print(f"  - Uncorrected:  {s['significant_uncorrected']}")
        print(f"  - Bonferroni:   {s['significant_bonferroni']}")
        print(f"  - FDR:          {s['significant_fdr']}")
        if s['mean_effect_size']:
            print(f"Mean effect size (Cohen's d): {s['mean_effect_size']:.3f}")
        print()
    
    def plot_results(
        self,
        report: AnalysisReport,
        output_dir: Optional[Path] = None,
        show: bool = True
    ) -> None:
        """Generate visualization plots."""
        if not HAS_MATPLOTLIB:
            warnings.warn("matplotlib not available, skipping plots")
            return
        
        # 1. Bar chart: LHS vs Control means
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Plot 1: Mean comparison bar chart
        ax = axes[0, 0]
        sentences = [c.sentence_id for c in report.comparisons]
        lhs_means = [c.lhs_mean for c in report.comparisons]
        ctrl_means = [c.control_mean for c in report.comparisons]
        lhs_stds = [c.lhs_std for c in report.comparisons]
        ctrl_stds = [c.control_std for c in report.comparisons]
        
        x = np.arange(len(sentences))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, lhs_means, width, yerr=lhs_stds, 
                       label='LHS (with operator)', color='steelblue', capsize=3)
        bars2 = ax.bar(x + width/2, ctrl_means, width, yerr=ctrl_stds,
                       label='Control', color='lightcoral', capsize=3)
        
        ax.set_xlabel('Sentence')
        ax.set_ylabel('Metric Value')
        ax.set_title('LHS vs Control: Mean Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(sentences)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        # Mark significant differences
        for i, comp in enumerate(report.comparisons):
            if comp.is_significant(use_corrected=False):
                ymax = max(comp.lhs_mean + comp.lhs_std, comp.control_mean + comp.control_std)
                ax.annotate('*', xy=(i, ymax * 1.05), ha='center', fontsize=14, fontweight='bold')
        
        # Plot 2: Effect sizes
        ax = axes[0, 1]
        effect_sizes = [c.cohens_d if c.cohens_d else 0 for c in report.comparisons]
        colors = ['green' if d > 0.5 else 'orange' if d > 0.2 else 'red' for d in effect_sizes]
        
        bars = ax.barh(sentences, effect_sizes, color=colors)
        ax.axvline(x=0.2, color='orange', linestyle='--', alpha=0.7, label='Small effect (0.2)')
        ax.axvline(x=0.5, color='yellow', linestyle='--', alpha=0.7, label='Medium effect (0.5)')
        ax.axvline(x=0.8, color='green', linestyle='--', alpha=0.7, label='Large effect (0.8)')
        ax.axvline(x=0, color='gray', linestyle='-', alpha=0.5)
        
        ax.set_xlabel("Cohen's d")
        ax.set_title('Effect Sizes')
        ax.legend(loc='lower right', fontsize=8)
        ax.grid(axis='x', alpha=0.3)
        
        # Plot 3: P-values comparison
        ax = axes[1, 0]
        p_uncorr = [c.t_pvalue if c.t_pvalue else 1.0 for c in report.comparisons]
        p_bonf = [c.bonferroni_p if c.bonferroni_p else 1.0 for c in report.comparisons]
        
        x = np.arange(len(sentences))
        width = 0.35
        
        ax.bar(x - width/2, [-np.log10(p) for p in p_uncorr], width,
               label='Uncorrected', color='blue', alpha=0.7)
        ax.bar(x + width/2, [-np.log10(p) for p in p_bonf], width,
               label='Bonferroni', color='red', alpha=0.7)
        
        ax.axhline(y=-np.log10(0.05), color='gray', linestyle='--', label='α=0.05')
        ax.set_xlabel('Sentence')
        ax.set_ylabel('-log₁₀(p)')
        ax.set_title('Statistical Significance')
        ax.set_xticks(x)
        ax.set_xticklabels(sentences)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        # Plot 4: Forest plot (confidence intervals)
        ax = axes[1, 1]
        y_positions = np.arange(len(sentences))
        
        for i, comp in enumerate(report.comparisons):
            mean_diff = comp.mean_diff if comp.mean_diff else (comp.lhs_mean - comp.control_mean)
            ci_lo = comp.ci_lower if comp.ci_lower else mean_diff
            ci_hi = comp.ci_upper if comp.ci_upper else mean_diff
            
            color = 'green' if ci_lo > 0 else 'red' if ci_hi < 0 else 'gray'
            ax.errorbar(mean_diff, i, xerr=[[mean_diff - ci_lo], [ci_hi - mean_diff]],
                       fmt='o', color=color, capsize=5, capthick=2, markersize=8)
        
        ax.axvline(x=0, color='gray', linestyle='-', alpha=0.5)
        ax.set_xlabel('Mean Difference (LHS - Control)')
        ax.set_ylabel('Sentence')
        ax.set_title('Forest Plot: 95% CI for Mean Difference')
        ax.set_yticks(y_positions)
        ax.set_yticklabels(sentences)
        ax.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            filepath = output_dir / 'apl_analysis_plots.png'
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            print(f"Plots saved to: {filepath}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def export_latex_table(self, report: AnalysisReport, filepath: Path) -> None:
        """Export results as LaTeX table."""
        lines = [
            r"\begin{table}[h]",
            r"\centering",
            r"\caption{APL Seven Sentences Test Results}",
            r"\label{tab:apl_results}",
            r"\begin{tabular}{lrrrrrr}",
            r"\toprule",
            r"Sentence & $\bar{x}_{LHS}$ & $\bar{x}_{Ctrl}$ & Cohen's $d$ & $p$-value & $p_{Bonf}$ & Sig. \\",
            r"\midrule"
        ]
        
        for comp in report.comparisons:
            d = f"{comp.cohens_d:.3f}" if comp.cohens_d else "---"
            p = f"{comp.t_pvalue:.4f}" if comp.t_pvalue else "---"
            pb = f"{comp.bonferroni_p:.4f}" if comp.bonferroni_p else "---"
            sig = "*" if comp.is_significant(use_corrected=True) else ""
            
            lines.append(
                f"{comp.sentence_id} & {comp.lhs_mean:.4f} & {comp.control_mean:.4f} & "
                f"{d} & {p} & {pb} & {sig} \\\\"
            )
        
        lines.extend([
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}"
        ])
        
        with open(filepath, 'w') as f:
            f.write('\n'.join(lines))
        
        print(f"LaTeX table saved to: {filepath}")
    
    def export_csv(self, report: AnalysisReport, filepath: Path) -> None:
        """Export results as CSV."""
        import csv
        
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'sentence_id', 'n_lhs', 'n_control',
                'lhs_mean', 'lhs_std', 'control_mean', 'control_std',
                'cohens_d', 't_statistic', 'p_value', 'bonferroni_p', 'fdr_p',
                'mean_diff', 'ci_lower', 'ci_upper', 'significant'
            ])
            
            for comp in report.comparisons:
                writer.writerow([
                    comp.sentence_id, comp.n_lhs, comp.n_control,
                    comp.lhs_mean, comp.lhs_std, comp.control_mean, comp.control_std,
                    comp.cohens_d, comp.t_statistic, comp.t_pvalue,
                    comp.bonferroni_p, comp.fdr_p,
                    comp.mean_diff, comp.ci_lower, comp.ci_upper,
                    comp.is_significant(use_corrected=True)
                ])
        
        print(f"CSV exported to: {filepath}")


# ============================================================================
# CLI
# ============================================================================

def main(argv: List[str] = None) -> int:
    parser = argparse.ArgumentParser(
        description='APL Test Results Analyzer',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        'files',
        nargs='+',
        type=Path,
        help='JSON result files to analyze'
    )
    parser.add_argument(
        '--alpha', '-a',
        type=float,
        default=0.05,
        help='Significance level (default: 0.05)'
    )
    parser.add_argument(
        '--plot', '-p',
        action='store_true',
        help='Generate visualization plots'
    )
    parser.add_argument(
        '--no-show',
        action='store_true',
        help='Do not display plots (save only)'
    )
    parser.add_argument(
        '--output', '-o',
        type=Path,
        default=Path('analysis'),
        help='Output directory for exports'
    )
    parser.add_argument(
        '--latex',
        action='store_true',
        help='Export LaTeX table'
    )
    parser.add_argument(
        '--csv',
        action='store_true',
        help='Export CSV'
    )
    
    args = parser.parse_args(argv)
    
    # Create analyzer
    analyzer = APLResultsAnalyzer(alpha=args.alpha)
    
    # Process each file
    for filepath in args.files:
        if not filepath.exists():
            print(f"Warning: File not found: {filepath}")
            continue
        
        print(f"\nAnalyzing: {filepath}")
        report = analyzer.load_results(filepath)
        analyzer.print_report(report)
        
        # Generate outputs
        args.output.mkdir(parents=True, exist_ok=True)
        
        if args.plot:
            analyzer.plot_results(report, args.output, show=not args.no_show)
        
        if args.latex:
            analyzer.export_latex_table(report, args.output / 'results_table.tex')
        
        if args.csv:
            analyzer.export_csv(report, args.output / 'results.csv')
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
