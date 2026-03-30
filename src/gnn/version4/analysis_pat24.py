import numpy as np

data        = np.load(r'F:\FORTH_Final_Thesis\FORTH-Thesis\gnn\version4\features\features_all.npz', allow_pickle=True)
X           = data['X']
y           = data['y']
patient_ids = data['patient_ids']
subject_ids = data['subject_ids']
feat_names  = data['feature_names'].tolist()

dtf_idx = feat_names.index('graph_dtf_asymmetry')

mask = (patient_ids == 'PAT24') & (y == 1)
subjs = subject_ids[mask]
vals  = X[mask, dtf_idx]

for s in np.unique(subjs):
    v = vals[subjs == s]
    print(f'Subject {s:02d}: mean DTF asymmetry ictal = {v.mean():+.4f}  (n={len(v)})')