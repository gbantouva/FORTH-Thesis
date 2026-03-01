import numpy as np
from pathlib import Path

# Check what files exist for subject_02
p = Path(r'F:\FORTH_Final_Thesis\FORTH-Thesis\preprocessed_epochs')
for f in sorted(p.glob('subject_02*')):
    print(f.name)

labels = np.load(r'F:\FORTH_Final_Thesis\FORTH-Thesis\preprocessed_epochs\subject_02_labels.npy')
print(labels.shape)
print(np.unique(labels, return_counts=True))