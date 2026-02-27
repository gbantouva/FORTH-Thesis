"""
Visualize Patient Information and Seizure Durations
====================================================
Creates comprehensive visualizations showing:
1. Subject → Patient mapping
2. Seizure durations for each subject
3. Pre-ictal, ictal, post-ictal periods
4. Patients with multiple seizures

Usage:
------
python visualize_patient_info.py \
    --json_file F:\FORTH_Final_Thesis\FORTH-Thesis\figures\preprocessing\patient_info.json \
    --output_dir F:\FORTH_Final_Thesis\FORTH-Thesis\figures\preprocessing
"""

import argparse
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
import seaborn as sns

def plot_patient_overview(data, output_dir):
    """
    Create comprehensive visualization of patient information.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    subjects = data['subjects']
    n_subjects = len(subjects)
    
    # =========================================================================
    # Extract data
    # =========================================================================
    
    subject_ids = [s['subject_id'] for s in subjects]
    patient_ids = [s['patient_id'] for s in subjects]
    seizure_durations = [s['seizure_timing']['duration_sec'] for s in subjects]
    pre_ictal_durations = [s['period_durations']['pre_ictal_sec'] for s in subjects]
    post_ictal_durations = [s['period_durations']['post_ictal_sec'] for s in subjects]
    total_durations = [s['signal_info']['total_duration_sec'] for s in subjects]
    
    # Create patient ID mapping (assign colors to unique patients)
    unique_patients = sorted(list(set(patient_ids)))
    patient_to_idx = {p: i for i, p in enumerate(unique_patients)}
    patient_colors = [patient_to_idx[p] for p in patient_ids]
    
    # =========================================================================
    # FIGURE 1: Complete Overview (4 subplots)
    # =========================================================================
    
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    # -------------------------------------------------------------------------
    # Plot 1: Subject → Patient Mapping (Color-coded)
    # -------------------------------------------------------------------------
    ax1 = fig.add_subplot(gs[0, :])
    
    # Create colormap
    n_unique = len(unique_patients)
    colors = plt.cm.tab20(np.linspace(0, 1, n_unique))
    
    # Plot bars
    bars = ax1.bar(subject_ids, [1]*n_subjects, color=[colors[pc] for pc in patient_colors],
                   edgecolor='black', linewidth=0.5)
    
    # Mark patients with multiple seizures
    patient_to_subjects = data.get('patient_to_subjects_mapping', {})
    multiple_seizure_patients = {p: s for p, s in patient_to_subjects.items() if len(s) > 1}
    
    for patient_id, subj_list in multiple_seizure_patients.items():
        for subj_id in subj_list:
            idx = subject_ids.index(subj_id)
            ax1.text(subj_id, 1.05, '★', ha='center', va='bottom', 
                    fontsize=12, color='red', fontweight='bold')
    
    ax1.set_xlabel('Subject ID', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Patient', fontsize=14, fontweight='bold')
    ax1.set_title('Subject → Patient Mapping (★ = Multiple Seizures from Same Patient)', 
                  fontsize=16, fontweight='bold')
    ax1.set_xlim(0, n_subjects + 1)
    ax1.set_ylim(0, 1.2)
    ax1.set_yticks([])
    ax1.grid(axis='x', alpha=0.3)
    
    # Add legend for patients with multiple seizures
    if multiple_seizure_patients:
        legend_text = "Patients with multiple seizures:\n"
        for patient_id, subj_list in sorted(multiple_seizure_patients.items())[:5]:  # Show first 5
            legend_text += f"{patient_id}: subjects {subj_list}\n"
        ax1.text(0.02, 0.98, legend_text, transform=ax1.transAxes,
                fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # -------------------------------------------------------------------------
    # Plot 2: Seizure Duration per Subject
    # -------------------------------------------------------------------------
    ax2 = fig.add_subplot(gs[1, :])
    
    bars = ax2.bar(subject_ids, seizure_durations, color='crimson', alpha=0.7, 
                   edgecolor='black', linewidth=0.5)
    
    # Add mean line
    mean_duration = np.mean(seizure_durations)
    ax2.axhline(mean_duration, color='blue', linestyle='--', linewidth=2, 
               label=f'Mean: {mean_duration:.1f}s')
    
    # Highlight shortest and longest
    min_idx = np.argmin(seizure_durations)
    max_idx = np.argmax(seizure_durations)
    bars[min_idx].set_color('green')
    bars[max_idx].set_color('purple')
    
    ax2.set_xlabel('Subject ID', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Seizure Duration (seconds)', fontsize=14, fontweight='bold')
    ax2.set_title('Seizure Duration per Subject', fontsize=16, fontweight='bold')
    ax2.grid(alpha=0.3, axis='y')
    ax2.legend(fontsize=10)
    
    # Add annotations for min and max
    ax2.text(min_idx + 1, seizure_durations[min_idx], 
            f'Min: {seizure_durations[min_idx]:.1f}s', 
            ha='center', va='bottom', fontsize=9, fontweight='bold', color='green')
    ax2.text(max_idx + 1, seizure_durations[max_idx], 
            f'Max: {seizure_durations[max_idx]:.1f}s', 
            ha='center', va='bottom', fontsize=9, fontweight='bold', color='purple')
    
    # -------------------------------------------------------------------------
    # Plot 3: Pre-ictal, Ictal, Post-ictal Stacked Bars
    # -------------------------------------------------------------------------
    ax3 = fig.add_subplot(gs[2, 0])
    
    # Create stacked bars
    ax3.bar(subject_ids, pre_ictal_durations, label='Pre-ictal', 
           color='steelblue', alpha=0.8, edgecolor='black', linewidth=0.3)
    ax3.bar(subject_ids, seizure_durations, bottom=pre_ictal_durations, 
           label='Ictal (Seizure)', color='crimson', alpha=0.8, edgecolor='black', linewidth=0.3)
    ax3.bar(subject_ids, post_ictal_durations, 
           bottom=[p + i for p, i in zip(pre_ictal_durations, seizure_durations)],
           label='Post-ictal', color='orange', alpha=0.8, edgecolor='black', linewidth=0.3)
    
    ax3.set_xlabel('Subject ID', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Time (seconds)', fontsize=12, fontweight='bold')
    ax3.set_title('Recording Period Breakdown', fontsize=14, fontweight='bold')
    ax3.legend(fontsize=10, loc='upper left')
    ax3.grid(alpha=0.3, axis='y')
    
    # -------------------------------------------------------------------------
    # Plot 4: Seizure Duration Distribution
    # -------------------------------------------------------------------------
    ax4 = fig.add_subplot(gs[2, 1])
    
    # Histogram
    n, bins, patches = ax4.hist(seizure_durations, bins=15, color='crimson', 
                                alpha=0.7, edgecolor='black', linewidth=1)
    
    # Add statistics
    stats_text = f"Min:    {np.min(seizure_durations):.1f}s\n"
    stats_text += f"Max:    {np.max(seizure_durations):.1f}s\n"
    stats_text += f"Mean:   {np.mean(seizure_durations):.1f}s\n"
    stats_text += f"Median: {np.median(seizure_durations):.1f}s\n"
    stats_text += f"Std:    {np.std(seizure_durations):.1f}s"
    
    ax4.text(0.65, 0.95, stats_text, transform=ax4.transAxes,
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Add vertical lines for mean and median
    ax4.axvline(np.mean(seizure_durations), color='blue', linestyle='--', 
               linewidth=2, label='Mean')
    ax4.axvline(np.median(seizure_durations), color='green', linestyle='--', 
               linewidth=2, label='Median')
    
    ax4.set_xlabel('Seizure Duration (seconds)', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Number of Subjects', fontsize=12, fontweight='bold')
    ax4.set_title('Distribution of Seizure Durations', fontsize=14, fontweight='bold')
    ax4.legend(fontsize=10)
    ax4.grid(alpha=0.3, axis='y')
    
    # -------------------------------------------------------------------------
    # Save
    # -------------------------------------------------------------------------
    plt.suptitle('TUC Dataset - Patient and Seizure Information Overview', 
                fontsize=18, fontweight='bold', y=0.995)
    
    plt.savefig(output_dir / 'patient_overview.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Saved: patient_overview.png")
    
    # =========================================================================
    # FIGURE 2: Detailed Timeline View (first 10 subjects)
    # =========================================================================
    
    fig, axes = plt.subplots(10, 1, figsize=(16, 20), sharex=True)
    
    for i in range(min(10, n_subjects)):
        ax = axes[i]
        subject = subjects[i]
        
        subject_id = subject['subject_id']
        patient_id = subject['patient_id']
        
        pre_ictal = subject['period_durations']['pre_ictal_sec']
        ictal = subject['period_durations']['ictal_sec']
        post_ictal = subject['period_durations']['post_ictal_sec']
        
        # Create timeline
        total = pre_ictal + ictal + post_ictal
        
        # Plot bars
        ax.barh(0, pre_ictal, left=0, height=0.5, color='steelblue', 
               edgecolor='black', label='Pre-ictal')
        ax.barh(0, ictal, left=pre_ictal, height=0.5, color='crimson', 
               edgecolor='black', label='Ictal')
        ax.barh(0, post_ictal, left=pre_ictal+ictal, height=0.5, color='orange', 
               edgecolor='black', label='Post-ictal')
        
        # Add labels
        ax.set_ylabel(f'S{subject_id:02d}\n{patient_id}', fontsize=10, fontweight='bold')
        ax.set_ylim(-0.5, 0.5)
        ax.set_xlim(0, max(total_durations))
        ax.grid(alpha=0.3, axis='x')
        
        # Add duration text
        ax.text(pre_ictal/2, 0, f'{pre_ictal:.0f}s', 
               ha='center', va='center', fontsize=8, fontweight='bold')
        ax.text(pre_ictal + ictal/2, 0, f'{ictal:.1f}s', 
               ha='center', va='center', fontsize=8, fontweight='bold', color='white')
        ax.text(pre_ictal + ictal + post_ictal/2, 0, f'{post_ictal:.0f}s', 
               ha='center', va='center', fontsize=8, fontweight='bold')
        
        if i == 0:
            ax.legend(loc='upper right', fontsize=9)
    
    axes[-1].set_xlabel('Time (seconds)', fontsize=12, fontweight='bold')
    plt.suptitle('Recording Timeline (First 10 Subjects)', fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'timeline_view.png', dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Saved: timeline_view.png")
    
    # =========================================================================
    # FIGURE 3: Patient-level Summary (if multiple seizures exist)
    # =========================================================================
    
    if multiple_seizure_patients:
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Prepare data
        patient_list = []
        seizure_counts = []
        mean_durations = []
        
        for patient_id in sorted(unique_patients):
            subject_list = patient_to_subjects.get(patient_id, [])
            patient_list.append(patient_id)
            seizure_counts.append(len(subject_list))
            
            # Get durations for this patient's seizures
            durations = [subjects[s-1]['seizure_timing']['duration_sec'] for s in subject_list]
            mean_durations.append(np.mean(durations))
        
        x = np.arange(len(patient_list))
        
        # Create grouped bar chart
        width = 0.35
        bars1 = ax.bar(x - width/2, seizure_counts, width, label='Number of Seizures',
                      color='steelblue', alpha=0.8, edgecolor='black')
        
        ax2 = ax.twinx()
        bars2 = ax2.bar(x + width/2, mean_durations, width, label='Mean Duration (s)',
                       color='crimson', alpha=0.8, edgecolor='black')
        
        # Highlight patients with multiple seizures
        for i, (patient_id, count) in enumerate(zip(patient_list, seizure_counts)):
            if count > 1:
                bars1[i].set_edgecolor('red')
                bars1[i].set_linewidth(3)
        
        ax.set_xlabel('Patient ID', fontsize=12, fontweight='bold')
        ax.set_ylabel('Number of Seizures', fontsize=12, fontweight='bold', color='steelblue')
        ax2.set_ylabel('Mean Seizure Duration (s)', fontsize=12, fontweight='bold', color='crimson')
        ax.set_title('Patient-Level Summary', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(patient_list, rotation=45, ha='right', fontsize=8)
        
        ax.tick_params(axis='y', labelcolor='steelblue')
        ax2.tick_params(axis='y', labelcolor='crimson')
        
        # Add legends
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=10)
        
        ax.grid(alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'patient_summary.png', dpi=200, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Saved: patient_summary.png")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize patient information and seizure durations"
    )
    parser.add_argument("--json_file", required=True,
                       help="Path to patient_info.json")
    parser.add_argument("--output_dir", required=True,
                       help="Output directory for plots")
    
    args = parser.parse_args()
    
    # Load JSON
    with open(args.json_file, 'r') as f:
        data = json.load(f)
    
    print("="*80)
    print("CREATING PATIENT INFORMATION VISUALIZATIONS")
    print("="*80)
    print(f"\nInput:  {args.json_file}")
    print(f"Output: {args.output_dir}")
    print(f"Subjects: {data['n_subjects']}")
    print("="*80)
    
    # Create plots
    plot_patient_overview(data, args.output_dir)
    
    print("\n" + "="*80)
    print("VISUALIZATION COMPLETE")
    print("="*80)
    print("\n📊 Created plots:")
    print("  • patient_overview.png    - Complete 4-panel overview")
    print("  • timeline_view.png       - Detailed timeline for first 10 subjects")
    if any(len(s) > 1 for s in data.get('patient_to_subjects_mapping', {}).values()):
        print("  • patient_summary.png     - Patient-level summary")
    print("="*80)


if __name__ == "__main__":
    main()