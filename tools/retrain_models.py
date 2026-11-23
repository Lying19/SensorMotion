"""Reproduce all three deployment models in one pinned environment.
Run from any directory: python tools/retrain_models.py --features /path/to/features.csv
The input features.csv must come from the original training dataset and contain trip/cls.
"""
import argparse
import json
import platform
from pathlib import Path
import sys
import joblib
import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'backend' / 'model_core'))
from common import normalize_per_trip


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--features', type=Path, default=ROOT / 'tools' / 'features.csv')
    args = p.parse_args()
    if not args.features.is_file():
        p.error(f'Cannot find features: {args.features}. Supply --features /path/to/features.csv')
    if sklearn.__version__ != '1.8.0':
        p.error(f'Expected scikit-learn==1.8.0, got {sklearn.__version__}; install requirements.txt first')
    df = pd.read_csv(args.features)
    cols = [c for c in df.columns if c.startswith(('acc_', 'gyr_'))]
    if len(cols) != 28 or not {'trip', 'cls'}.issubset(df.columns):
        p.error(f'Expected 28 acc_/gyr_ features and trip/cls; found {len(cols)} features')
    norm = normalize_per_trip(df, cols)
    X = norm[cols]
    y = norm['cls']
    if X.isna().any().any() or not np.isfinite(X.to_numpy()).all():
        p.error('Training features contain NaN or infinity')
    rng = np.random.default_rng(0)
    sampled = rng.choice(len(norm), min(8000, len(norm)), replace=False)
    models = {
        'RandomForest': (RandomForestClassifier(300, class_weight='balanced', n_jobs=-1, random_state=0), X, y, 'final_model.joblib'),
        'SVM': (make_pipeline(StandardScaler(), SVC(C=3, class_weight='balanced', probability=True, random_state=0)), X.iloc[sampled], y.iloc[sampled], 'svm_model.joblib'),
        'kNN': (make_pipeline(StandardScaler(), KNeighborsClassifier(15)), X.iloc[sampled], y.iloc[sampled], 'knn_model.joblib'),
    }
    out = ROOT / 'backend' / 'models'
    out.mkdir(parents=True, exist_ok=True)
    manifest = {'python': platform.python_version(), 'scikit_learn': sklearn.__version__, 'numpy': np.__version__, 'pandas': pd.__version__, 'scipy': scipy.__version__, 'joblib': joblib.__version__, 'total_windows': len(norm), 'feature_count': len(cols), 'normalization': 'per-trip median/IQR', 'models': {}}
    for name, (model, x, labels, filename) in models.items():
        print(f'Training {name}: {len(x)} windows', flush=True)
        model.fit(x, labels)
        bundle = {'model': model, 'cols': cols, 'classes': list(model.classes_), 'training_windows': len(x), 'normalization': 'per-trip median/IQR', 'sklearn_version': sklearn.__version__}
        target = out / filename
        joblib.dump(bundle, target, compress=3)
        check = joblib.load(target)
        if list(check['model'].classes_) != list(model.classes_):
            raise RuntimeError(f'Roundtrip failed for {name}')
        print(f'Saved {target.name} ({target.stat().st_size / 1024**2:.1f} MiB)', flush=True)
        manifest['models'][name] = {'file': filename, 'training_windows': len(x), 'classes': list(model.classes_), 'size_bytes': target.stat().st_size}
    (out / 'model_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    print('All models trained, saved and reload-checked.', flush=True)

if __name__ == '__main__':
    main()
