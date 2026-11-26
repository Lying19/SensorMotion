"""Train deployment bundles from the verified W15 features, not evaluation folds."""
import hashlib
import importlib
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import sklearn
import scipy
import sys


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def main():
    c = importlib.import_module('common')
    ev = importlib.import_module('03_train_evaluate')
    sc = importlib.import_module('05b_scaler_comparison')
    root = Path(c.OUT_DIR)
    source = root / '07a_extended_window_sweep' / 'W15'
    f = source / 'features.csv'
    q = source / '視窗品質.csv'
    actual = {'features': digest(f), 'quality': digest(q)}
    if actual != read_json(source / 'features_manifest.json'):
        raise ValueError('W15 feature manifest mismatch')
    prior = root / '07b_final_window_sweep' / 'run_manifest.json'
    sig = read_json(prior)
    if sig['reused_predictions_and_features']['W15/features.csv'] != actual['features']:
        raise ValueError('W15 differs from phase 07b')
    prep = Path(__file__).with_name('02_preprocess_features.py')
    original = sig['phase07a']['phase07']
    if digest(prep) != original['preprocess']:
        raise ValueError('Preprocessing source changed')
    for name in ('common.py', '03_train_evaluate.py', '05b_scaler_comparison.py'):
        if digest(Path(__file__).with_name(name)) != original['baseline']['files'][name]:
            raise ValueError(f'Source changed: {name}')
    if (c.FS, c.OVERLAP, c.TRIM_SEC, c.MAX_GAP) != (50, .5, 30, .5):
        raise ValueError('Unexpected preprocessing settings')
    D = pd.read_csv(f)
    suffixes = 'mean std max min rms skew kurt zcr energy domf spec_ent p_low p_mid p_high'.split()
    cols = [f'{p}_{s}' for p in ('acc','gyr') for s in suffixes]
    if [x for x in D if x.startswith(('acc_', 'gyr_'))] != cols:
        raise ValueError('Unexpected feature order')
    meta = ['trip','cls','position','device']
    if len(D)!=5718 or D.trip.nunique()!=168 or D[meta].isna().any().any() or not np.isfinite(D[cols].to_numpy()).all():
        raise ValueError('Unexpected W15 data')
    if (D.groupby('trip')[meta[1:]].nunique()!=1).any().any() or set(D.cls)!=set(ev.LABELS):
        raise ValueError('Invalid trip metadata')
    base = pd.read_csv(root/'features.csv')
    expected = base[meta].drop_duplicates().query("trip != '1019_MRT_hand'").sort_values('trip').reset_index(drop=True)
    if not D[meta].drop_duplicates().sort_values('trip').reset_index(drop=True).equals(expected):
        raise ValueError('Unexpected W15 cohort')
    Q = pd.read_csv(q)
    missing = Q[Q.trip=='1019_MRT_hand']
    if len(missing)!=1 or missing.iloc[0].valid_windows!=0 or missing.iloc[0].status!='無有效視窗':
        raise ValueError('Unexpected missing trip quality')
    config = dict(fs=50, window_sec=15, window_samples=750, overlap=.5, step_samples=375, trim_sec=30, max_gap_sec=.5, sensors=['Accelerometer','Gyroscope'], magnitude=True, per_trip_normalization=False, alignment='independent_resampling_truncate_to_shorter', aggregation='majority_vote', feature_order=cols)
    versions = dict(python=sys.version.split()[0], sklearn=sklearn.__version__, numpy=np.__version__, pandas=pd.__version__, scipy=scipy.__version__, joblib=joblib.__version__)
    out = root/'08_final_models'; out.mkdir(parents=True, exist_ok=True)
    manifest = dict(schema_version=1, source_sha256=actual, phase07b_manifest_sha256=digest(prior), script_sha256=digest(__file__), preprocessing_sha256=digest(prep), versions=versions, preprocessing=config, training_windows=len(D), training_trips=D.trip.nunique(), excluded_trips=['1019_MRT_hand'], models={})
    path = out/'run_manifest.json'
    if path.exists():
        old=read_json(path)
        if {k:v for k,v in old.items() if k!='models'} != {k:v for k,v in manifest.items() if k!='models'}:
            raise ValueError('Existing final model manifest differs; rename output folder before rerunning')
    X=D[cols].to_numpy(); y=D.cls.to_numpy()
    for key,name,method,filename in [('RF','RandomForest','N0','final_model.joblib'),('SVM','SVM(RBF)','N4','svm_model.joblib'),('kNN','kNN(k=15)','N4','knn_model.joblib')]:
        print(f'Training {key}: {len(X)} windows',flush=True)
        model=sc.build_model(ev,name,method).fit(X,y)
        bundle=dict(model=model, cols=cols, classes=model.classes_.tolist(), model_name=key, scaler=sc.METHODS[method], preprocessing=config, versions=versions, training_windows=len(X), training_trips=D.trip.nunique(), source_sha256=actual['features'])
        dest=out/filename; tmp=out/(filename+'.tmp')
        joblib.dump(bundle,tmp,compress=3)
        restored=joblib.load(tmp)
        if not np.array_equal(model.predict(X[:32]),restored['model'].predict(X[:32])):
            raise RuntimeError('Serialization check failed')
        tmp.replace(dest)
        manifest['models'][key]=dict(file=filename,sha256=digest(dest),classes=bundle['classes'],scaler=bundle['scaler'],supports_predict_proba=hasattr(model,'predict_proba'))
    tmp=path.with_suffix('.tmp'); tmp.write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8'); tmp.replace(path)
    print('Completed:',out)

if __name__=='__main__':
    main()
