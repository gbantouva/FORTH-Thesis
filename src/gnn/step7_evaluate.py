"""
Step 7 - Evaluation & Results Summary
======================================
Reads the LOSO results JSON from step 6 and produces:
  1. Per-subject results table (console + CSV)
  2. Aggregated metrics table
  3. Per-subject bar chart
  4. ROC-style metric summary figure

This script also serves as the template for comparing models later
(Baseline GCN → GAT → more complex models).

Usage:
  python step7_evaluate.Apy \
      --results_dir results/baseline_gcn \
      --model_name "Baseline GCN"

  # Compare multiple models:
  python step7_evaluate.py \
      --results_dir results/baseline_gcn results/gat \
      --model_name "Baseline GCN" "GAT"
"""

import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


# ============================================================================
# LOAD & FORMAT
# ============================================================================

def load_results(results_dir):
    path = Path(results_dir) / "loso_results.json"
    with open(path) as f:
        return json.load(f)


def fold_results_to_df(results):
    rows = []
    for fold in results['fold_results']:
        rows.append({
            'Subject':     fold['test_subject'],
            'N Epochs':    fold['n_test'],
            'Accuracy':    fold['accuracy'],
            'F1':          fold['f1'],
            'AUC':         fold['auc'],
            'Sensitivity': fold['sensitivity'],
            'Specificity': fold['specificity'],
        })
    return pd.DataFrame(rows).set_index('Subject')


# ============================================================================
# SINGLE MODEL REPORT
# ============================================================================

def print_report(results, model_name, output_dir):
    output_dir = Path(output_dir)

    df = fold_results_to_df(results)
    macro = results['macro_metrics']
    micro = results['micro_metrics']

    # Console table
    print("\n" + "=" * 80)
    print(f"MODEL: {model_name}")
    print("=" * 80)
    print(df.to_string(float_format=lambda x: f"{x:.3f}"))

    print("\n─── Macro-average (mean ± std across subjects) ───")
    for metric in ['accuracy', 'f1', 'auc', 'sensitivity', 'specificity']:
        mu  = macro[f'{metric}_mean']
        std = macro[f'{metric}_std']
        print(f"  {metric:15s}: {mu:.3f} ± {std:.3f}")

    print("\n─── Micro-average (all epochs pooled) ───")
    for metric in ['accuracy', 'f1', 'auc', 'sensitivity', 'specificity']:
        print(f"  {metric:15s}: {micro[metric]:.3f}")

    # Save CSV
    df.to_csv(output_dir / 'per_subject_results.csv')
    print(f"\n✅ Per-subject CSV saved")

    # Plot 1: Per-subject bars
    _plot_per_subject(df, model_name, output_dir)

    # Plot 2: Summary radar/bar chart
    _plot_summary(macro, model_name, output_dir)

    return df, macro, micro


def _plot_per_subject(df, model_name, output_dir):
    metrics = ['Accuracy', 'F1', 'Sensitivity', 'Specificity']
    n_subj  = len(df)
    x       = np.arange(n_subj)
    width   = 0.2

    fig, ax = plt.subplots(figsize=(max(14, n_subj * 0.5), 6))
    colors  = ['#4e79a7', '#f28e2b', '#59a14f', '#e15759']

    for i, (metric, color) in enumerate(zip(metrics, colors)):
        offset = (i - len(metrics) / 2 + 0.5) * width
        ax.bar(x + offset, df[metric], width, label=metric,
               color=color, alpha=0.8, edgecolor='black', linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels([f"S{s:02d}" for s in df.index], rotation=45, ha='right')
    ax.set_ylabel('Score')
    ax.set_title(f'{model_name} — Per-Subject LOSO Results', fontweight='bold')
    ax.set_ylim(0, 1.1)
    ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5, label='Chance')
    ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left')
    ax.grid(alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(output_dir / 'per_subject_bars.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("✅ Per-subject bar chart saved")


def _plot_summary(macro, model_name, output_dir):
    metrics = ['accuracy', 'f1', 'auc', 'sensitivity', 'specificity']
    labels  = ['Accuracy', 'F1', 'AUC', 'Sensitivity', 'Specificity']
    means   = [macro[f'{m}_mean'] for m in metrics]
    stds    = [macro[f'{m}_std']  for m in metrics]

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ['#4e79a7', '#f28e2b', '#59a14f', '#e15759', '#76b7b2']
    bars = ax.bar(labels, means, yerr=stds, capsize=5,
                  color=colors, alpha=0.8, edgecolor='black')

    for bar, mean in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.02,
                f'{mean:.3f}', ha='center', fontsize=10, fontweight='bold')

    ax.set_ylim(0, 1.2)
    ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5, label='Chance')
    ax.set_ylabel('Score (mean ± std)', fontweight='bold')
    ax.set_title(f'{model_name} — Aggregated LOSO Metrics', fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(output_dir / 'summary_metrics.png', dpi=200)
    plt.close()
    print("✅ Summary metrics chart saved")


# ============================================================================
# MULTI-MODEL COMPARISON
# ============================================================================

def compare_models(results_dirs, model_names, output_dir):
    """
    Compare multiple models side by side.
    Useful when you add GAT or more complex models later.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_macros = []
    for rdir, name in zip(results_dirs, model_names):
        res = load_results(rdir)
        all_macros.append((name, res['macro_metrics']))

    metrics = ['accuracy', 'f1', 'auc', 'sensitivity', 'specificity']
    labels  = ['Accuracy', 'F1', 'AUC', 'Sensitivity', 'Specificity']
    n_models = len(all_macros)
    x        = np.arange(len(metrics))
    width    = 0.8 / n_models

    fig, ax = plt.subplots(figsize=(12, 6))
    palette = ['#4e79a7', '#f28e2b', '#59a14f', '#e15759',
               '#76b7b2', '#ff9da7', '#9c755f']

    for i, (name, macro) in enumerate(all_macros):
        means = [macro[f'{m}_mean'] for m in metrics]
        stds  = [macro[f'{m}_std']  for m in metrics]
        offset = (i - n_models / 2 + 0.5) * width
        ax.bar(x + offset, means, width, yerr=stds, capsize=4,
               label=name, color=palette[i % len(palette)],
               alpha=0.85, edgecolor='black', linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel('Score (mean ± std across subjects)', fontweight='bold')
    ax.set_title('Model Comparison — LOSO Cross-Validation', fontweight='bold')
    ax.axhline(0.5, color='gray', linestyle='--', alpha=0.4, label='Chance')
    ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left')
    ax.grid(alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(output_dir / 'model_comparison.png', dpi=200, bbox_inches='tight')
    plt.close()

    # Table
    print("\n" + "=" * 80)
    print("MODEL COMPARISON TABLE")
    print("=" * 80)
    header = f"{'Model':<25}" + "".join(f"{m:>18}" for m in labels)
    print(header)
    print("-" * 80)
    for name, macro in all_macros:
        row = f"{name:<25}"
        for m in metrics:
            mu  = macro[f'{m}_mean']
            std = macro[f'{m}_std']
            row += f"  {mu:.3f}±{std:.3f}  "
        print(row)
    print("=" * 80)
    print(f"\n✅ Comparison chart saved to {output_dir}/model_comparison.png")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Evaluate GNN results")
    parser.add_argument("--results_dir", nargs='+', required=True,
                        help="One or more results directories")
    parser.add_argument("--model_name",  nargs='+', required=True,
                        help="Model names (same order as results_dir)")
    parser.add_argument("--output_dir",  default=None,
                        help="Where to save comparison plots "
                             "(default: first results_dir)")
    args = parser.parse_args()

    if len(args.results_dir) != len(args.model_name):
        raise ValueError("--results_dir and --model_name must have same length")

    output_dir = Path(args.output_dir or args.results_dir[0])
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("STEP 7 — EVALUATION")
    print("=" * 80)

    if len(args.results_dir) == 1:
        # Single model report
        results = load_results(args.results_dir[0])
        print_report(results, args.model_name[0], output_dir)
    else:
        # Multi-model comparison
        for rdir, name in zip(args.results_dir, args.model_name):
            results = load_results(rdir)
            print_report(results, name, Path(rdir))

        compare_models(args.results_dir, args.model_name, output_dir)


if __name__ == "__main__":
    main()
