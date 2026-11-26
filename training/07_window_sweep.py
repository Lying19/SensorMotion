"""07: compare 1/2/5 s unified models; reuse verified phase-06 E1 predictions."""
import argparse, hashlib, importlib, json, os, subprocess, sys
from pathlib import Path
import numpy as np
import pandas as pd
import sklearn
import scipy

def digest(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def check_manifest(path, value):
    if path.exists() and json.loads(path.read_text(encoding='utf-8')) != value:
        raise ValueError(f'Inputs changed: rename output directory containing {path}')
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--window',type=int,choices=[1,2,5]); ap.add_argument('--model',choices=['RF','RF_N5','SVM'])
    args=ap.parse_args()
    c=importlib.import_module('common'); ev=importlib.import_module('03_train_evaluate'); sc=importlib.import_module('05b_scaler_comparison'); core=importlib.import_module('06_core_experiments')
    root=Path(c.OUT_DIR).resolve(); out=root/'07_window_sweep'; out.mkdir(parents=True,exist_ok=True)
    base=root/'features.csv'; prior=root/'06_core_experiments'; prep=Path(__file__).with_name('02_preprocess_features.py')
    versions={'sklearn':sklearn.__version__,'numpy':np.__version__,'pandas':pd.__version__}
    files=[base,Path(core.__file__),Path(ev.__file__),Path(sc.__file__),Path(c.__file__)]
    expected={'files':{p.name:digest(p) for p in files},'versions':versions}
    if json.loads((prior/'run_manifest.json').read_text(encoding='utf-8'))!=expected:
        raise ValueError('Phase 06 manifest mismatch; cannot reuse W2 safely. Restore original files/environment.')
    if (c.FS,c.WIN_SEC,c.OVERLAP,c.TRIM_SEC,c.MAX_GAP)!=(50,2,.5,30,.5): raise ValueError('Unexpected baseline settings')
    # Fingerprint the actual raw sensor sources, including extracted and ZIP inputs.
    raw={}
    for r in c.find_recordings():
        paths=[Path(r['archive'])] if r['archive'] is not None else [Path(r['folder'])/n for n in ('Accelerometer.csv','Gyroscope.csv','Metadata.csv')]
        for p in paths:
            if str(p.resolve()) not in raw: raw[str(p.resolve())]=digest(p)
    sig={'baseline':expected,'raw':raw,'preprocess':digest(prep),'script':digest(__file__),'scipy':scipy.__version__,'phase06_predictions':{k:digest(prior/f'{k}_E1_all_predictions.csv') for k in core.MODELS}}
    check_manifest(out/'run_manifest.json',sig)
    B=pd.read_csv(base); cols=[x for x in B if x.startswith(('acc_','gyr_'))]; meta=['trip','cls','position','device']
    ref=B[meta].drop_duplicates().sort_values('trip').reset_index(drop=True)
    if len(ref)!=169 or len(B)!=45902 or len(cols)!=28: raise ValueError('Unexpected W2 baseline')
    rows=[]; distributions=[]; cms=[]
    for w in (1,2,5):
        folder=out/f'W{w}'; folder.mkdir(exist_ok=True); source=base if w==2 else folder/'features.csv'
        if w!=2 and not (folder/'features_manifest.json').exists():
            if args.window not in (None,w): continue
            subprocess.run([sys.executable,str(prep)],env={**os.environ,'WIN_SEC':str(w),'OUT_DIR':str(folder),'DATA_DIR':str(Path(c.DATA_DIR).resolve())},check=True)
            check_manifest(folder/'features_manifest.json',{'features':digest(source),'quality':digest(folder/'視窗品質.csv')})
        if w!=2:
            check_manifest(folder/'features_manifest.json',{'features':digest(source),'quality':digest(folder/'視窗品質.csv')})
        D=pd.read_csv(source); D['row_id']=np.arange(len(D))
        if [x for x in D if x.startswith(('acc_','gyr_'))]!=cols or not np.isfinite(D[cols].to_numpy()).all() or not D[meta].drop_duplicates().sort_values('trip').reset_index(drop=True).equals(ref):
            raise ValueError(f'W{w}: features or trip cohort differs; investigate quality report before comparison')
        distributions.append(D.groupby(['position','cls']).agg(趟數=('trip','nunique'),視窗數=('trip','size')).reset_index().assign(視窗秒數=w))
        Z=c.normalize_per_trip(D,cols)
        if not np.isfinite(Z[cols].to_numpy()).all(): raise ValueError('Non-finite N5 features')
        for key,(name,method) in core.MODELS.items():
            path=folder/f'{key}_predictions.csv'
            if w==2:
                old=pd.read_csv(prior/f'{key}_E1_all_predictions.csv')
                if not old[['row_id']+meta].equals(D[['row_id']+meta]) or not old.pred.isin(ev.LABELS).all(): raise ValueError('Invalid W2 predictions')
                core.save(old,path)
            elif not path.exists() and (args.window not in (None,w) or args.model not in (None,key)): continue
            model=sc.build_model(ev,name,'N0' if method=='N5' else method)
            T=core.evaluate(Z if method=='N5' else D,cols,model,key,ev,'E1',path)
            V=core.trips(T); core.save(V,folder/f'{key}_每趟結果.csv')
            rows.append(dict(core.summarize(T,ev,key,'E1','all'),視窗秒數=w,分類器=name,方法=sc.METHODS[method],來源='Phase06' if w==2 else 'New LORO'))
            ev.plot_cm(T.cls.to_numpy(),T.pred.to_numpy(),f'W{w} {key}',folder/f'{key}_混淆矩陣.png')
            from sklearn.metrics import confusion_matrix
            for level,y,p in [('window',T.cls,T.pred),('trip',V.cls,V.pred)]:
                cm=confusion_matrix(y,p,labels=ev.LABELS)
                for i,a in enumerate(ev.LABELS):
                    for j,b in enumerate(ev.LABELS): cms.append(dict(視窗秒數=w,模型=key,層級=level,實際=a,預測=b,數量=cm[i,j]))
    R=pd.DataFrame(rows); core.save(R,out/'結果_視窗長度比較.csv'); core.save(pd.concat(distributions,ignore_index=True),out/'資料分布.csv'); core.save(pd.DataFrame(cms),out/'混淆矩陣數值.csv')
    Q=R.merge(R[R.視窗秒數==2][['模型']+core.METRICS],on='模型',suffixes=('','_W2'),validate='many_to_one')
    for m in core.METRICS: Q[m+'_差異百分點']=(Q[m]-Q[m+'_W2'])*100
    core.save(Q,out/'與2秒比較.csv')
    for m in core.METRICS[:4]: core.save(R.sort_values(m,ascending=False),out/f'排名_{m}.csv')
    print(R.round(4).to_string(index=False)); print(f'Completed {len(R)}/9 combinations: {out}')
if __name__=='__main__': main()
