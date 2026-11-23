"""Reproducible deployment training: same normalized 28 features as final RF."""
from pathlib import Path
import sys
import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import normalize_per_trip

BASE = Path(__file__).resolve().parent.parent
D = pd.read_csv(BASE / 'features.csv')
cols = [c for c in D.columns if c.startswith(('acc_', 'gyr_'))]
Z = normalize_per_trip(D, cols)
X, y = Z[cols], Z['cls']
# Same MAX_TRAIN=8000 used in teammate's SVM/kNN LORO training.
rng = np.random.default_rng(0)
idx = rng.choice(len(Z), min(8000, len(Z)), replace=False)
models = {
 'SVM': make_pipeline(StandardScaler(), SVC(C=3, class_weight='balanced', probability=True, random_state=0)),
 'kNN': make_pipeline(StandardScaler(), KNeighborsClassifier(15))
}
for name, model in models.items():
 print('Training', name, 'on',len(idx),'windows', flush=True)
 model.fit(X.iloc[idx], y.iloc[idx])
 bundle = {'model':model, 'cols':cols, 'classes':list(model.classes_), 'training_windows':len(idx), 'normalization':'per-trip median/IQR', 'source':'03_train_evaluate.py; deployment fit with MAX_TRAIN=8000'}
 joblib.dump(bundle, BASE / 'backend' / 'models' / ('svm_model.joblib' if name=='SVM' else 'knn_model.joblib'), compress=3)
 print(name, model.predict(X.iloc[:2]), flush=True)
