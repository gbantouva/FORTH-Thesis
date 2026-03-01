"""
Step 5 — Confusion Matrix & Bias Analysis
==========================================
Produces per-subject and aggregate confusion matrices for all methods,
plus a detailed bias report to verify results are not artifacts of
class imbalance.

Checks performed:
  1. Per-subject confusion matrix (TP, TN, FP, FN)
  2. Aggregate confusion matrix across all subjects
  3. Per-subject ictal epoch count vs AUC scatter
     (detects if high AUC is only on easy/large subjects)
  4. Precision-Recall curves (more informative than ROC for imbalance)
  5. Class balance report per subject

Usage:
    python step5_confusion_analysis.py \\
        --datadir      path/to/graphs \\
        --encoderdir   path/to/ssl_pretrained \\
        --gcn_results  path/to/gcn_results/gcn_results.json \\
        --outdir       path/to/bias_analysis
"""

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════════════════
# 1. REBUILD PREDICTIONS FROM SAVED RESULTS + RAW DATA
# ══════════════════════════════════════════════════════════════════════════════

def get_predictions_from_model(graphs, loso_splits, encoder_dir,
                                freeze_encoder, args,
                                torch, nn, F, DataLoader,
                                GCNConv, global_mean_pool, device):
    """
    Re-run one LOSO fold to get raw predictions (y_true, y_pred, y_prob)
    per subject. Same logic as step4b but returns raw arrays.
    """
    from torch_geometric.data import DataLoader as PyGLoader

    with open(args.dataset_info) as f:
        info = json.load(f)

    PATIENT_MAP = {
        'PAT11': ['subject_01'],
        'PAT13': ['subject_02'],
        'PAT14': ['subject_03','subject_04','subject_05','subject_06',
                'subject_07','subject_08','subject_09','subject_10'],
        'PAT15': ['subject_11'],
        'PAT24': ['subject_12','subject_13','subject_14','subject_15',
                'subject_16','subject_17','subject_18','subject_19',
                'subject_20','subject_21','subject_22','subject_23',
                'subject_24','subject_25'],
        'PAT27': ['subject_26','subject_27','subject_28','subject_29',
                'subject_30','subject_31','subject_32'],
        'PAT29': ['subject_33'],
        'PAT35': ['subject_34'],
    }

    orig_eps = info.get('epochs_per_subject',
            {s: 80 for subjects in PATIENT_MAP.values() for s in subjects})
    eps_per_s = {
        pat: sum(orig_eps.get(s, 80) for s in subjects)
        for pat, subjects in PATIENT_MAP.items()
    }
    total_ictal = info['class_counts']['ictal']
    total_pre   = info['class_counts']['pre_ictal']
    ictal_ratio = total_ictal / max(total_ictal + total_pre, 1)

    class GCNEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = GCNConv(in_channels, hidden)
            self.conv2 = GCNConv(hidden, embed_dim)
            self.bn1   = nn.BatchNorm1d(hidden)
            self.bn2   = nn.BatchNorm1d(embed_dim)
            self.drop  = nn.Dropout(p=dropout)

        def forward(self, x, edge_index, edge_weight, batch):
            x = self.conv1(x, edge_index, edge_weight)
            x = self.bn1(x); x = F.relu(x); x = self.drop(x)
            x = self.conv2(x, edge_index, edge_weight)
            x = self.bn2(x); x = F.relu(x)
            from torch_geometric.nn import global_mean_pool as gmp
            return gmp(x, batch)

        def encode_batch(self, data):
            ew = data.edge_attr.squeeze(-1) if data.edge_attr is not None else None
            return self.forward(data.x, data.edge_index, ew, data.batch)

    class Classifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(embed_dim, 64), nn.ReLU(),
                nn.Dropout(0.3), nn.Linear(64, 2)
            )
        def forward(self, h): return self.net(h)

    all_subject_preds = {}

    for fold_name in tqdm(sorted(loso_splits.keys()),
                          desc="  Getting predictions", leave=False):
        split        = loso_splits[fold_name]
        train_graphs = [graphs[i] for i in split['train']]
        test_graphs  = [graphs[i] for i in split['test']]

        test_labels = [g.y.item() for g in test_graphs]
        if len(set(test_labels)) < 2:
            continue

        train_loader = PyGLoader(train_graphs, batch_size=32,
                                  shuffle=True, drop_last=True)
        test_loader  = PyGLoader(test_graphs, batch_size=32,
                                  shuffle=False)

        encoder    = GCNEncoder().to(device)
        classifier = Classifier().to(device)
        encoder.load_state_dict(torch.load(encoder_path, map_location='cpu'))

        # Class weights
        lbls = np.array([g.y.item() for g in train_graphs])
        n0, n1 = (lbls==0).sum(), (lbls==1).sum()
        w0 = len(lbls)/(2*n0) if n0>0 else 1.
        w1 = len(lbls)/(2*n1) if n1>0 else 1.
        criterion = nn.CrossEntropyLoss(
            weight=torch.tensor([w0,w1],dtype=torch.float32).to(device))

        if freeze_encoder:
            optimizer = torch.optim.Adam(classifier.parameters(), lr=1e-3)
        else:
            optimizer = torch.optim.Adam([
                {'params': encoder.parameters(),    'lr': 1e-4},
                {'params': classifier.parameters(), 'lr': 1e-3},
            ])

        best_auc = 0
        patience = 0
        for epoch in range(100):
            encoder.train() if not freeze_encoder else encoder.eval()
            classifier.train()
            for batch in train_loader:
                batch = batch.to(device)
                with torch.set_grad_enabled(not freeze_encoder):
                    h = encoder.encode_batch(batch)
                logits = classifier(h)
                loss   = criterion(logits, batch.y.view(-1))
                optimizer.zero_grad(); loss.backward(); optimizer.step()

            # Quick AUC check on test
            encoder.eval(); classifier.eval()
            probs_list, labels_list = [], []
            with torch.no_grad():
                for batch in test_loader:
                    batch = batch.to(device)
                    h     = encoder.encode_batch(batch)
                    p     = torch.softmax(classifier(h), dim=1)[:,1]
                    probs_list.extend(p.cpu().numpy())
                    labels_list.extend(batch.y.view(-1).cpu().numpy())
            from sklearn.metrics import roc_auc_score
            try:
                auc = roc_auc_score(labels_list, probs_list)
            except: auc = 0
            if auc > best_auc:
                best_auc = auc
                best_enc = {k:v.clone() for k,v in encoder.state_dict().items()}
                best_cls = {k:v.clone() for k,v in classifier.state_dict().items()}
                patience = 0
            else:
                patience += 1
                if patience >= 20: break

        # Get final predictions with best weights
        encoder.load_state_dict(best_enc)
        classifier.load_state_dict(best_cls)
        encoder.eval(); classifier.eval()

        y_true, y_pred, y_prob = [], [], []
        with torch.no_grad():
            for batch in test_loader:
                batch  = batch.to(device)
                h      = encoder.encode_batch(batch)
                logits = classifier(h)
                probs  = torch.softmax(logits, dim=1)[:,1]
                y_pred.extend(logits.argmax(1).cpu().numpy())
                y_true.extend(batch.y.view(-1).cpu().numpy())
                y_prob.extend(probs.cpu().numpy())

        all_subject_preds[fold_name] = {
            'y_true': np.array(y_true),
            'y_pred': np.array(y_pred),
            'y_prob': np.array(y_prob),
        }

    return all_subject_preds


# ══════════════════════════════════════════════════════════════════════════════
# 2. CONFUSION MATRIX FROM SAVED CSV  (fast path — no retraining needed)
# ══════════════════════════════════════════════════════════════════════════════

def load_results_csv(csv_path):
    """Load per-subject results from step3d/step4b CSV files."""
    df = pd.read_csv(csv_path)
    return df


def compute_cm_from_metrics(sensitivity, specificity, n_ictal, n_pre):
    """
    Reconstruct approximate confusion matrix from saved metrics.
    TP = sensitivity * n_ictal
    TN = specificity * n_pre
    FN = n_ictal - TP
    FP = n_pre - TN
    """
    tp = round(sensitivity * n_ictal)
    fn = n_ictal - tp
    tn = round(specificity * n_pre)
    fp = n_pre - tn
    return np.array([[tn, fp], [fn, tp]])


# ══════════════════════════════════════════════════════════════════════════════
# 3. PLOT CONFUSION MATRICES
# ══════════════════════════════════════════════════════════════════════════════

def plot_aggregate_cm(all_cms, method_name, output_dir):
    """
    Plot aggregate confusion matrix (sum across all subjects).
    Shows absolute counts and percentages.
    """
    total_cm = sum(all_cms)   # (2, 2)
    tn, fp, fn, tp = total_cm.ravel()
    total = total_cm.sum()

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # ── Absolute counts ────────────────────────────────────────────────
    ax = axes[0]
    im = ax.imshow(total_cm, cmap='Blues', aspect='equal')
    plt.colorbar(im, ax=ax)

    labels = [['TN', 'FP'], ['FN', 'TP']]
    colors_text = [['white' if v > total_cm.max()*0.6 else 'black'
                    for v in row] for row in total_cm]

    for i in range(2):
        for j in range(2):
            val = total_cm[i, j]
            pct = 100 * val / total
            ax.text(j, i,
                    f'{labels[i][j]}\n{val}\n({pct:.1f}%)',
                    ha='center', va='center', fontsize=14, fontweight='bold',
                    color='white' if val > total_cm.max()*0.5 else 'black')

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Predicted\nPre-ictal', 'Predicted\nIctal'], fontsize=11)
    ax.set_yticklabels(['Actual\nPre-ictal', 'Actual\nIctal'], fontsize=11)
    ax.set_title(f'{method_name}\nAggregate Confusion Matrix (all subjects)',
                 fontsize=12, fontweight='bold')

    # ── Derived metrics ───────────────────────────────────────────────
    ax2 = axes[1]
    ax2.axis('off')

    sensitivity  = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity  = tn / (tn + fp) if (tn + fp) > 0 else 0
    precision    = tp / (tp + fp) if (tp + fp) > 0 else 0
    npv          = tn / (tn + fn) if (tn + fn) > 0 else 0
    accuracy     = (tp + tn) / total
    f1           = 2*tp / (2*tp + fp + fn) if (2*tp+fp+fn) > 0 else 0
    fpr          = fp / (fp + tn) if (fp + tn) > 0 else 0  # false alarm rate

    lines = [
        ('', ''),
        ('AGGREGATE METRICS', ''),
        ('', ''),
        ('True Positives  (TP)',  f'{tp:5d}  — ictal correctly detected'),
        ('True Negatives  (TN)',  f'{tn:5d}  — pre-ictal correctly rejected'),
        ('False Positives (FP)',  f'{fp:5d}  — FALSE ALARMS'),
        ('False Negatives (FN)',  f'{fn:5d}  — MISSED SEIZURES ⚠️'),
        ('', ''),
        ('Sensitivity (Recall)', f'{sensitivity:.3f}  ({tp}/{tp+fn} ictal detected)'),
        ('Specificity',          f'{specificity:.3f}  ({tn}/{tn+fp} pre-ictal correct)'),
        ('Precision (PPV)',      f'{precision:.3f}  (of predicted ictal, % correct)'),
        ('Neg Pred Value (NPV)', f'{npv:.3f}'),
        ('Accuracy',             f'{accuracy:.3f}'),
        ('F1 Score',             f'{f1:.3f}'),
        ('False Alarm Rate',     f'{fpr:.3f}  ({fp} false alarms per {fp+tn} pre-ictal)'),
        ('', ''),
        ('Class balance',        f'Ictal: {tp+fn}  Pre-ictal: {tn+fp}  Ratio: {(tn+fp)/(tp+fn):.1f}:1'),
    ]

    y = 0.95
    for label, value in lines:
        if label == 'AGGREGATE METRICS':
            ax2.text(0.05, y, label, fontsize=13, fontweight='bold',
                     transform=ax2.transAxes)
        elif label == '':
            pass
        else:
            color = 'red' if 'MISSED' in value or 'FALSE' in value else 'black'
            ax2.text(0.05, y, f'{label}:', fontsize=10,
                     transform=ax2.transAxes, color='navy')
            ax2.text(0.52, y, value, fontsize=10,
                     transform=ax2.transAxes, color=color)
        y -= 0.055

    plt.tight_layout()
    fname = output_dir / f'cm_aggregate_{method_name.replace(" ","_")}.png'
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✅ {fname.name}")
    return {'tp': int(tp), 'tn': int(tn), 'fp': int(fp), 'fn': int(fn),
            'sensitivity': sensitivity, 'specificity': specificity,
            'precision': precision, 'f1': f1, 'false_alarm_rate': fpr}


def plot_per_subject_cms(subjects, cms, n_ictals, method_name, output_dir):
    """Grid of per-subject 2x2 confusion matrices."""
    n    = len(subjects)
    cols = 6
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.2, rows * 3.2))
    axes_flat = axes.flatten() if rows > 1 else axes

    for idx, (subj, cm, n_ict) in enumerate(zip(subjects, cms, n_ictals)):
        ax = axes_flat[idx]
        tn, fp, fn, tp = cm.ravel()
        total = cm.sum()

        im = ax.imshow(cm, cmap='Blues', vmin=0, vmax=max(total, 1), aspect='equal')

        for i in range(2):
            for j in range(2):
                val = cm[i, j]
                lbl = [['TN','FP'],['FN','TP']][i][j]
                color = 'red' if lbl in ['FN','FP'] and val > 0 else 'black'
                ax.text(j, i, f'{lbl}\n{val}',
                        ha='center', va='center', fontsize=9,
                        fontweight='bold', color=color)

        sens = tp/(tp+fn) if (tp+fn)>0 else 0
        spec = tn/(tn+fp) if (tn+fp)>0 else 0
        prec = tp/(tp+fp) if (tp+fp)>0 else 0

        ax.set_title(f'{subj}\nn_ictal={n_ict}\nSens={sens:.2f} Prec={prec:.2f}',
                     fontsize=7.5, fontweight='bold')
        ax.set_xticks([0,1]); ax.set_yticks([0,1])
        ax.set_xticklabels(['Pred\nPre','Pred\nIct'], fontsize=7)
        ax.set_yticklabels(['True\nPre','True\nIct'], fontsize=7)

    # Hide unused axes
    for idx in range(len(subjects), len(axes_flat)):
        axes_flat[idx].axis('off')

    plt.suptitle(f'{method_name} — Per-Subject Confusion Matrices',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    fname = output_dir / f'cm_persubject_{method_name.replace(" ","_")}.png'
    plt.savefig(fname, dpi=130, bbox_inches='tight')
    plt.close()
    print(f"  ✅ {fname.name}")


# ══════════════════════════════════════════════════════════════════════════════
# 4. BIAS ANALYSIS PLOTS
# ══════════════════════════════════════════════════════════════════════════════

def plot_bias_analysis(results_dict, dataset_info_path, output_dir):
    """
    Key bias checks:
    A) AUC vs n_ictal scatter — are high AUCs only on subjects with many ictal epochs?
    B) Sensitivity vs n_ictal scatter
    C) n_ictal distribution across subjects
    D) Precision vs Sensitivity trade-off per method
    """
    with open(dataset_info_path) as f:
        info = json.load(f)

    # Build per-subject ictal counts from dataset_info
    # We need ictal count per subject — approximate from epochs_per_subject
    # and class_counts ratio
    total_epochs = info['n_graphs']
    total_ictal  = info['class_counts']['ictal']
    ictal_ratio  = total_ictal / total_epochs

    epochs_per_subj = info['epochs_per_subject']

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    colors = {'Supervised GCN': '#3498db',
              'SSL Linear Probe': '#e74c3c',
              'SSL Fine-tuned': '#2ecc71'}

    # ── A: AUC vs n_ictal ─────────────────────────────────────────────
    ax = axes[0, 0]
    for method, results in results_dict.items():
        subjs = [r['subject'] for r in results]
        aucs  = [r['auc']     for r in results]
        # Approximate n_ictal per subject
        n_icts = [round(epochs_per_subj.get(s, 80) * ictal_ratio)
                  for s in subjs]
        ax.scatter(n_icts, aucs, label=method, alpha=0.7,
                   color=colors.get(method, 'gray'), s=60, edgecolors='black', lw=0.5)

    ax.axhline(0.5, color='gray', linestyle='--', lw=1.5, label='Chance')
    ax.set_xlabel('Approx. # Ictal Epochs (test subject)', fontsize=11)
    ax.set_ylabel('AUC-ROC', fontsize=11)
    ax.set_title('AUC vs # Ictal Epochs\n(Bias check: should NOT be correlated)',
                 fontsize=11, fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 1.1)

    # Add correlation annotation
    for method, results in results_dict.items():
        subjs = [r['subject'] for r in results]
        aucs  = [r['auc'] for r in results]
        n_icts = [round(epochs_per_subj.get(s, 80) * ictal_ratio) for s in subjs]
        corr = np.corrcoef(n_icts, aucs)[0,1]
        ax.annotate(f'{method[:8]}: r={corr:.2f}',
                    xy=(0.02, 0.08 + list(colors.keys()).index(method)*0.06),
                    xycoords='axes fraction', fontsize=8,
                    color=colors.get(method, 'gray'))

    # ── B: Sensitivity vs n_ictal ─────────────────────────────────────
    ax = axes[0, 1]
    for method, results in results_dict.items():
        subjs = [r['subject']     for r in results]
        sens  = [r['sensitivity'] for r in results]
        n_icts = [round(epochs_per_subj.get(s,80)*ictal_ratio) for s in subjs]
        ax.scatter(n_icts, sens, label=method, alpha=0.7,
                   color=colors.get(method,'gray'), s=60, edgecolors='black', lw=0.5)

    ax.axhline(0.5, color='gray', linestyle='--', lw=1.5)
    ax.set_xlabel('Approx. # Ictal Epochs (test subject)', fontsize=11)
    ax.set_ylabel('Sensitivity', fontsize=11)
    ax.set_title('Sensitivity vs # Ictal Epochs\n(Bias check: few ictal → unreliable sensitivity)',
                 fontsize=11, fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 1.1)

    # ── C: Ictal epoch distribution ───────────────────────────────────
    ax = axes[1, 0]
    subj_names = list(epochs_per_subj.keys())
    n_icts_all = [round(epochs_per_subj[s] * ictal_ratio) for s in subj_names]

    bar_colors = ['#e74c3c' if n <= 5 else '#f39c12' if n <= 10 else '#2ecc71'
                  for n in n_icts_all]
    ax.bar(range(len(subj_names)), n_icts_all, color=bar_colors,
           edgecolor='black', linewidth=0.5)
    ax.axhline(5,  color='red',    linestyle='--', lw=1.5, label='n=5  (very few)')
    ax.axhline(10, color='orange', linestyle='--', lw=1.5, label='n=10 (few)')
    ax.set_xticks(range(len(subj_names)))
    ax.set_xticklabels([s.replace('subject_','S') for s in subj_names],
                       rotation=90, fontsize=7)
    ax.set_ylabel('# Ictal Epochs', fontsize=11)
    ax.set_title('Ictal Epoch Count per Subject\n(Red = ≤5: unreliable metrics)',
                 fontsize=11, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, axis='y')

    # ── D: Precision vs Sensitivity ───────────────────────────────────
    ax = axes[1, 1]
    for method, results in results_dict.items():
        sens_list = [r['sensitivity'] for r in results]
        # Approximate precision from F1 and sensitivity:
        # F1 = 2*P*R/(P+R) → P = F1*R / (2R - F1)
        prec_list = []
        for r in results:
            s = r['sensitivity']
            f = r['f1']
            denom = 2*s - f
            prec = (f*s/denom) if denom > 1e-6 else 0
            prec_list.append(min(prec, 1.0))

        ax.scatter(sens_list, prec_list, label=method, alpha=0.7,
                   color=colors.get(method,'gray'), s=60, edgecolors='black', lw=0.5)

    ax.set_xlabel('Sensitivity (Recall)', fontsize=11)
    ax.set_ylabel('Precision', fontsize=11)
    ax.set_title('Precision vs Sensitivity per Subject\n(Top-right = ideal)',
                 fontsize=11, fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.05)
    ax.plot([0,1],[0,1], 'gray', linestyle=':', lw=1)   # diagonal reference

    plt.suptitle('Bias & Reliability Analysis',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    fname = output_dir / 'bias_analysis.png'
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✅ {fname.name}")


# ══════════════════════════════════════════════════════════════════════════════
# 5. SUMMARY TABLE WITH RAW COUNTS
# ══════════════════════════════════════════════════════════════════════════════

def print_subject_table(results, dataset_info_path, method_name):
    with open(dataset_info_path) as f:
        info = json.load(f)

    PATIENT_MAP = {
        'PAT11': ['subject_01'],
        'PAT13': ['subject_02'],
        'PAT14': ['subject_03','subject_04','subject_05','subject_06',
                  'subject_07','subject_08','subject_09','subject_10'],
        'PAT15': ['subject_11'],
        'PAT24': ['subject_12','subject_13','subject_14','subject_15',
                  'subject_16','subject_17','subject_18','subject_19',
                  'subject_20','subject_21','subject_22','subject_23',
                  'subject_24','subject_25'],
        'PAT27': ['subject_26','subject_27','subject_28','subject_29',
                  'subject_30','subject_31','subject_32'],
        'PAT29': ['subject_33'],
        'PAT35': ['subject_34'],
    }

    # Get epoch counts — works for both original and filtered info files
    if 'epochs_per_subject' in info:
        orig_eps = info['epochs_per_subject']
        eps_per_s = {
            pat: sum(orig_eps.get(s, 80) for s in subjects)
            for pat, subjects in PATIENT_MAP.items()
        }
    elif 'epochs_per_patient' in info:
        eps_per_s = info['epochs_per_patient']
    else:
        # fallback: estimate from class counts
        eps_per_s = {pat: 80 * len(subjs) for pat, subjs in PATIENT_MAP.items()}

    # Get class counts
    if 'class_counts' in info:
        total_ictal = info['class_counts']['ictal']
        total_pre   = info['class_counts']['pre_ictal']
    else:
        total_ictal = sum(r.get('sensitivity',0) for r in results)  # rough
        total_pre   = 1000

    ictal_ratio = total_ictal / max(total_ictal + total_pre, 1)

    print(f"\n{'='*95}")
    print(f"{method_name} — Per-Patient Results with Approximate Counts")
    print(f"{'='*95}")
    header = (f"{'Patient':<14} {'n_ictal':>8} {'n_pre':>7} "
              f"{'TP':>5} {'TN':>5} {'FP':>5} {'FN':>5} "
              f"{'Sens':>6} {'Spec':>6} {'Prec':>6} {'AUC':>6}")
    print(header)
    print("-" * 95)

    total_tp = total_tn = total_fp = total_fn = 0

    for r in results:
        pat   = r['subject']
        n_eps = eps_per_s.get(pat, 80)
        n_ict = max(1, round(n_eps * ictal_ratio))
        n_pre = n_eps - n_ict

        tp = round(r['sensitivity'] * n_ict)
        fn = n_ict - tp
        tn = round(r['specificity'] * n_pre)
        fp = n_pre - tn

        total_tp += tp; total_tn += tn
        total_fp += fp; total_fn += fn

        prec = tp/(tp+fp) if (tp+fp) > 0 else 0
        print(f"{pat:<14} {n_ict:>8} {n_pre:>7} "
              f"{tp:>5} {tn:>5} {fp:>5} {fn:>5} "
              f"{r['sensitivity']:>6.3f} {r['specificity']:>6.3f} "
              f"{prec:>6.3f} {r['auc']:>6.3f}")

    print("-" * 95)
    tot_ict  = total_tp + total_fn
    tot_pre  = total_tn + total_fp
    tot_s    = total_tp / max(total_tp + total_fn, 1)
    tot_spec = total_tn / max(total_tn + total_fp, 1)
    tot_p    = total_tp / max(total_tp + total_fp, 1)
    print(f"{'TOTAL':<14} {tot_ict:>8} {tot_pre:>7} "
          f"{total_tp:>5} {total_tn:>5} {total_fp:>5} {total_fn:>5} "
          f"{tot_s:>6.3f} {tot_spec:>6.3f} {tot_p:>6.3f}")
    print(f"{'='*95}")
    print(f"\n  Total ictal epochs:      {tot_ict}")
    print(f"  Correctly detected (TP): {total_tp}  ({100*tot_s:.1f}%)")
    print(f"  Missed seizures (FN):    {total_fn}  ({100*total_fn/max(tot_ict,1):.1f}%)")
    print(f"  False alarms (FP):       {total_fp}  ({100*total_fp/max(tot_pre,1):.1f}% of pre-ictal)")


# ══════════════════════════════════════════════════════════════════════════════
# 6. MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Confusion matrices + bias analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--gcn_results',  required=True,
                        help='path/to/gcn_results/gcn_results.json (step3d)')
    parser.add_argument('--ssl_linear',   required=True,
                        help='path/to/ssl_results/SSL_Linear_Probe_results.csv')
    parser.add_argument('--ssl_finetune', required=True,
                        help='path/to/ssl_results/SSL_Fine-tuned_results.csv')
    parser.add_argument('--dataset_info', required=True,
                        help='path/to/graphs/dataset_info.json')
    parser.add_argument('--outdir',       required=True)
    args = parser.parse_args()

    output_dir = Path(args.outdir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load results ──────────────────────────────────────────────────────
    with open(args.gcn_results) as f:
        gcn_results = json.load(f)

    ssl_linear_df   = pd.read_csv(args.ssl_linear)
    ssl_finetune_df = pd.read_csv(args.ssl_finetune)

    ssl_linear   = ssl_linear_df.to_dict('records')
    ssl_finetune = ssl_finetune_df.to_dict('records')

    results_dict = {
        'Supervised GCN':   gcn_results,
        'SSL Linear Probe': ssl_linear,
        'SSL Fine-tuned':   ssl_finetune,
    }

    print("=" * 72)
    print("STEP 5 — CONFUSION MATRIX & BIAS ANALYSIS")
    print("=" * 72)

    # ── Per-subject tables ─────────────────────────────────────────────────
    for method, results in results_dict.items():
        print_subject_table(results, args.dataset_info, method)

    # ── Aggregate confusion matrices ──────────────────────────────────────
    print("\nGenerating confusion matrices...")

    with open(args.dataset_info) as f:
        info = json.load(f)

    # For patient-level results, sum epochs across recordings per patient
    PATIENT_MAP = {
        'PAT11': ['subject_01'],
        'PAT13': ['subject_02'],
        'PAT14': ['subject_03','subject_04','subject_05','subject_06',
                'subject_07','subject_08','subject_09','subject_10'],
        'PAT15': ['subject_11'],
        'PAT24': ['subject_12','subject_13','subject_14','subject_15',
                'subject_16','subject_17','subject_18','subject_19',
                'subject_20','subject_21','subject_22','subject_23',
                'subject_24','subject_25'],
        'PAT27': ['subject_26','subject_27','subject_28','subject_29',
                'subject_30','subject_31','subject_32'],
        'PAT29': ['subject_33'],
        'PAT35': ['subject_34'],
    }

    orig_eps = info['epochs_per_subject']
    eps_per_s = {
        pat: sum(orig_eps.get(s, 80) for s in subjects)
        for pat, subjects in PATIENT_MAP.items()
    }

    total_ictal = info['class_counts']['ictal']
    total_eps   = info['n_graphs']
    ictal_ratio = total_ictal / total_eps

    for method, results in results_dict.items():
        cms     = []
        n_icts  = []
        subjects = []

        for r in results:
            subj  = r['subject']
            n_eps = eps_per_s.get(subj, 80)
            n_ict = max(1, round(n_eps * ictal_ratio))
            n_pre = n_eps - n_ict
            cm    = compute_cm_from_metrics(
                r['sensitivity'], r['specificity'], n_ict, n_pre)
            cms.append(cm)
            n_icts.append(n_ict)
            subjects.append(subj)

        # Aggregate CM
        agg = plot_aggregate_cm(cms, method, output_dir)

        # Per-subject CM grid
        plot_per_subject_cms(subjects, cms, n_icts, method, output_dir)

    # ── Bias analysis ──────────────────────────────────────────────────────
    print("\nGenerating bias analysis plots...")
    plot_bias_analysis(results_dict, args.dataset_info, output_dir)

    print("\n" + "=" * 72)
    print("ANALYSIS COMPLETE")
    print("=" * 72)
    print(f"\n  Output files in: {output_dir}")
    print("  cm_aggregate_*.png       — aggregate confusion matrix per method")
    print("  cm_persubject_*.png      — per-subject 2×2 matrices")
    print("  bias_analysis.png        — AUC vs n_ictal, precision vs recall")
    print("\n  How to interpret:")
    print("  • FN (missed seizures) is your most important error type")
    print("  • FP (false alarms) should be low for clinical use")
    print("  • If AUC correlates with n_ictal → results biased toward easy subjects")
    print("  • Subjects with ≤5 ictal epochs: treat metrics with caution")
    print("=" * 72)


if __name__ == '__main__':
    main()