"""SensorMotion inference. CLI: python training/predict.py RECORDING [--model all] [--mode live].
Import Predictor for Flask. Only load trusted local joblib files.
Live inputs must be cumulative snapshots of ONE session, in seconds_elapsed units.
"""
import argparse
import hashlib
import json
import sys
import zipfile
from pathlib import Path
from collections import Counter
import joblib
import numpy as np
import pandas as pd
import scipy
from scipy import signal
import sklearn

FS, WIN, STEP, MAX_GAP, TRIM_SEC = 50, 750, 375, .5, 30
FIELDS = ['seconds_elapsed','x','y','z']
LABELS = {'MRT':'捷運','bus':'公車','car':'汽車'}
MODELS = {'RF':'final_model.joblib','SVM':'svm_model.joblib','kNN':'knn_model.joblib'}
COLS = [f'{p}_{s}' for p in ('acc','gyr') for s in 'mean std max min rms skew kurt zcr energy domf spec_ent p_low p_mid p_high'.split()]
DEFAULT_DIR = Path(__file__).resolve().parent/'models'

def mag_resampled(df, trim):
    df = pd.DataFrame(df)[FIELDS].astype(float)
    t = df.seconds_elapsed.to_numpy()
    if not np.isfinite(df.to_numpy()).all() or (np.diff(t)<=0).any():
        raise ValueError('感測器數值或時間戳異常：必須有限且嚴格遞增')
    if len(t)<2: return np.array([]), np.array([],dtype=bool)
    m = np.sqrt(df.x**2+df.y**2+df.z**2).values
    grid = np.arange(t[0]+trim,t[-1]-trim,1/FS)
    out = np.interp(grid,t,m)
    idx = np.searchsorted(t,grid).clip(1,len(t)-1)
    return out, (t[idx]-t[idx-1])<=MAX_GAP

def feats(x, prefix):
    mean = x.mean(); xc = x - mean
    f, P = signal.welch(xc, FS, nperseg=len(xc)); Pn = P / (P.sum() + 1e-12)
    d = dict(mean=mean, std=xc.std(), max=x.max(), min=x.min(), rms=np.sqrt((xc**2).mean()),
             skew=pd.Series(xc).skew(), kurt=pd.Series(xc).kurt(), zcr=((xc[:-1]*xc[1:]) < 0).mean(),
             energy=(xc**2).mean(), domf=f[Pn.argmax()], spec_ent=-(Pn*np.log(Pn+1e-12)).sum(),
             p_low=Pn[f <= 3].sum(), p_mid=Pn[(f > 3) & (f <= 10)].sum(), p_high=Pn[f > 10].sum())
    return {f"{prefix}_{k}": v for k, v in d.items()}

def extract_windows(acc, gyro, mode='trip'):
    if mode not in ('trip','live'): raise ValueError('mode must be trip or live')
    trim = TRIM_SEC if mode=='trip' else 0
    ma,va=mag_resampled(acc,trim); mg,vg=mag_resampled(gyro,trim)
    n=min(len(ma),len(mg)); v=va[:n]&vg[:n]
    rows=[]; ids=[]; gaps=0; nonfinite=0
    total=max(0,(n-WIN)//STEP+1)
    for i,s in enumerate(range(0,n-WIN+1,STEP)):
        if not v[s:s+WIN].all(): gaps+=1; continue
        r=feats(ma[s:s+WIN],'acc'); r.update(feats(mg[s:s+WIN],'gyr'))
        if not np.isfinite(list(r.values())).all(): nonfinite+=1; continue
        rows.append(r); ids.append(i)
    q=dict(total_windows=total,valid_windows=len(rows),invalid_windows=total-len(rows),gap_windows=gaps,nonfinite_windows=nonfinite,valid_pct=round(100*len(rows)/total,1) if total else 0.0,acc_samples=len(ma),gyro_samples=len(mg),paired_samples=n,trim_sec=trim,latest_window_valid=bool(ids and ids[-1]==total-1),status='OK' if rows else ('insufficient_data' if not total else 'no_valid_windows'))
    return pd.DataFrame(rows,columns=COLS),ids,q

def read_recording(path):
    path=Path(path)
    names=['Accelerometer.csv','Gyroscope.csv']
    if path.is_dir(): return tuple(pd.read_csv(path/n,usecols=FIELDS) for n in names)
    with zipfile.ZipFile(path) as z:
        matches=[[p for p in z.namelist() if Path(p).name==n and not p.startswith('__MACOSX/')] for n in names]
        if any(len(m)!=1 for m in matches) or Path(matches[0][0]).parent!=Path(matches[1][0]).parent:
            raise ValueError('ZIP 必須只含同一趟錄製的兩個感測器 CSV')
        return tuple(pd.read_csv(z.open(m[0]),usecols=FIELDS) for m in matches)

def windows_of(folder, mode='trip'):
    return extract_windows(*read_recording(folder),mode=mode)[0]

class Predictor:
    def __init__(self, model_dir=DEFAULT_DIR, models=('RF','SVM','kNN')):
        root=Path(model_dir); manifest=json.loads((root/'run_manifest.json').read_text(encoding='utf-8-sig'))
        cfg=manifest['preprocessing']
        expected=dict(fs=FS,window_sec=15,window_samples=WIN,overlap=.5,step_samples=STEP,trim_sec=TRIM_SEC,max_gap_sec=MAX_GAP,sensors=['Accelerometer','Gyroscope'],magnitude=True,per_trip_normalization=False,alignment='independent_resampling_truncate_to_shorter',aggregation='majority_vote',feature_order=COLS)
        if cfg!=expected: raise ValueError('模型前處理規格不一致')
        versions=dict(sklearn=sklearn.__version__,numpy=np.__version__,pandas=pd.__version__,scipy=scipy.__version__,joblib=joblib.__version__)
        for k,v in versions.items():
            if manifest['versions'][k]!=v: raise RuntimeError(f'{k}: installed {v}, training {manifest["versions"][k]}')
        self.models={}
        for key in models:
            info=manifest['models'][key]; path=root/MODELS[key]
            if info['file']!=path.name or hashlib.sha256(path.read_bytes()).hexdigest()!=info['sha256']: raise ValueError(f'{key}: model checksum mismatch')
            b=joblib.load(path)
            if b['preprocessing']!=cfg or b['cols']!=COLS or b['model_name']!=key or b['source_sha256']!=manifest['source_sha256']['features'] or b['scaler']!=info['scaler'] or b['classes']!=info['classes'] or list(b['model'].classes_)!=b['classes'] or set(b['classes'])!=set(LABELS) or b['model'].n_features_in_!=28: raise ValueError(f'{key}: invalid bundle')
            self.models[key]=b['model']
    def predict(self, acc, gyro, mode='trip', model='SVM'):
        keys=list(self.models) if model=='all' else [model]
        if not keys or any(k not in self.models for k in keys): raise ValueError('requested model not loaded')
        W,ids,q=extract_windows(acc,gyro,mode)
        result=dict(mode=mode,quality=q,models={})
        for key in keys:
            pred=self.models[key].predict(W.to_numpy()).tolist() if len(W) else []
            counts=Counter(pred); votes={c:counts[c] for c in LABELS}
            # Same pandas majority-vote rule as evaluation, including ties.
            winner=str(pd.Series(pred).value_counts().index[0]) if pred else None
            latest=pred[-1] if pred and q['latest_window_valid'] else None
            result['models'][key]=dict(prediction=winner,label=LABELS.get(winner),latest_prediction=latest,latest_label=LABELS.get(latest),vote_counts=votes,vote_ratios={c:v/len(pred) if pred else 0.0 for c,v in votes.items()},tie=bool(pred and sum(v==max(votes.values()) for v in votes.values())>1),windows=[dict(window_index=i,start_offset_sec=i*STEP/FS,prediction=p) for i,p in zip(ids,pred)])
        return result
    def predict_recording(self,path,mode='trip',model='SVM'):
        return self.predict(*read_recording(path),mode=mode,model=model)

def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('recordings',nargs='+'); p.add_argument('--model',choices=['RF','SVM','kNN','all'],default='SVM'); p.add_argument('--mode',choices=['trip','live'],default='trip'); p.add_argument('--model-dir',type=Path,default=DEFAULT_DIR)
    args=p.parse_args(); engine=Predictor(args.model_dir,tuple(MODELS) if args.model=='all' else (args.model,)); failed=False
    for path in args.recordings:
        try: result=engine.predict_recording(path,args.mode,args.model); failed |= result['quality']['status']!='OK'
        except (ValueError,KeyError,OSError,zipfile.BadZipFile) as e: result={'error':str(e)}; failed=True
        print(json.dumps(dict(recording=path,**result),ensure_ascii=False,allow_nan=False))
    return int(failed)

if __name__=='__main__': sys.exit(main())
