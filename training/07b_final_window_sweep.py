"""07b: extend unified model comparison to 20/30 s; reuse phase-07a predictions."""
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
    ap.add_argument('--window',type=int,choices=[20,30]); ap.add_argument('--model',choices=['RF','SVM'])
    args=ap.parse_args()
    c=importlib.import_module('common'); ev=importlib.import_module('03_train_evaluate'); sc=importlib.import_module('05b_scaler_comparison'); core=importlib.import_module('06_core_experiments')
    root=Path(c.OUT_DIR).resolve(); previous=root/'07_window_sweep'; extended=root/'07a_extended_window_sweep'; out=root/'07b_final_window_sweep'; out.mkdir(parents=True,exist_ok=True)
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
    old_sig=json.loads((previous/'run_manifest.json').read_text(encoding='utf-8'))
    sig['script']=digest(Path(__file__).with_name('07_window_sweep.py'))
    if sig!=old_sig:
        raise ValueError('Phase 07 manifest mismatch: restore original raw data, scripts and environment.')
    reused={}
    for w in (1,2,5):
        f=previous/f'W{w}'
        if w!=2:
            actual={'features':digest(f/'features.csv'),'quality':digest(f/'視窗品質.csv')}
            if actual!=json.loads((f/'features_manifest.json').read_text(encoding='utf-8')):
                raise ValueError(f'Phase 07 W{w} feature manifest mismatch')
        for key in ('RF','SVM'):
            q=f/f'{key}_predictions.csv'; reused[f'W{w}/{q.name}']=digest(q)
            if w==2 and not pd.read_csv(q).equals(pd.read_csv(prior/f'{key}_E1_all_predictions.csv')):
                raise ValueError('W2 prediction hash mismatch')
    old_a=json.loads((extended/'run_manifest.json').read_text(encoding='utf-8'))
    expected_a={'phase07':old_sig,'phase07_manifest':digest(previous/'run_manifest.json'),'reused_predictions':reused,'script':digest(Path(__file__).with_name('07a_extended_window_sweep.py'))}
    if old_a!=expected_a: raise ValueError('Phase 07a manifest mismatch')
    reused_a={}
    for w in (1,2,5,10,15):
        f=extended/f'W{w}'
        if w in (10,15):
            actual={'features':digest(f/'features.csv'),'quality':digest(f/'視窗品質.csv')}
            if actual!=json.loads((f/'features_manifest.json').read_text(encoding='utf-8')): raise ValueError(f'Phase 07a W{w} feature manifest mismatch')
            reused_a[f'W{w}/features.csv']=actual['features']
            reused_a[f'W{w}/視窗品質.csv']=actual['quality']
        for key in ('RF','SVM'):
            q=f/f'{key}_predictions.csv'; reused_a[f'W{w}/{q.name}']=digest(q)
            if w in (1,2,5) and not pd.read_csv(q).equals(pd.read_csv(previous/f'W{w}'/q.name)):
                raise ValueError(f'Phase 07a W{w} {key} differs from phase 07')
    current={'phase07a_manifest':digest(extended/'run_manifest.json'),'phase07a':old_a,'reused_predictions_and_features':reused_a,'script':digest(__file__)}
    check_manifest(out/'run_manifest.json',current)
    B=pd.read_csv(base); cols=[x for x in B if x.startswith(('acc_','gyr_'))]; meta=['trip','cls','position','device']
    ref=B[meta].drop_duplicates().sort_values('trip').reset_index(drop=True)
    if len(ref)!=169 or len(B)!=45902 or len(cols)!=28: raise ValueError('Unexpected W2 baseline')
    rows=[]; distributions=[]; cms=[]; missing_rows=[]; predictions={}; cohorts={}
    for w in (1,2,5,10,15,20,30):
        folder=out/f'W{w}'; folder.mkdir(exist_ok=True); source=base if w==2 else (previous/f'W{w}'/'features.csv' if w in (1,5) else ((extended/f'W{w}'/'features.csv') if w in (10,15) else folder/'features.csv'))
        if w in (20,30) and not (folder/'features_manifest.json').exists():
            if args.window not in (None,w): continue
            subprocess.run([sys.executable,str(prep)],env={**os.environ,'WIN_SEC':str(w),'OUT_DIR':str(folder),'DATA_DIR':str(Path(c.DATA_DIR).resolve())},check=True)
            check_manifest(folder/'features_manifest.json',{'features':digest(source),'quality':digest(folder/'視窗品質.csv')})
        if w in (20,30):
            check_manifest(folder/'features_manifest.json',{'features':digest(source),'quality':digest(folder/'視窗品質.csv')})
        D=pd.read_csv(source); D['row_id']=np.arange(len(D))
        actual=D[meta].drop_duplicates().sort_values('trip').reset_index(drop=True)
        absent=set(ref.trip)-set(actual.trip)
        if [x for x in D if x.startswith(('acc_','gyr_'))]!=cols or not np.isfinite(D[cols].to_numpy()).all() or not actual.equals(ref[~ref.trip.isin(absent)].reset_index(drop=True)):
            raise ValueError(f'W{w}: invalid features or changed trip metadata')
        if w <= 15 and absent != ({'1019_MRT_hand'} if w==15 else set()):
            raise ValueError(f'W{w}: unexpected missing trips: {sorted(absent)}')
        if absent:
            quality=pd.read_csv((extended/f'W{w}' if w==15 else folder)/'視窗品質.csv')
            for trip in sorted(absent):
                q=quality[quality.trip==trip]
                if len(q)!=1 or q.iloc[0].valid_windows!=0 or q.iloc[0].status not in ('無有效視窗','裁切後太短') or not q[meta].reset_index(drop=True).equals(ref[ref.trip==trip].reset_index(drop=True)):
                    raise ValueError(f'W{w}: missing trip quality evidence invalid: {trip}')
                missing_rows.append(dict(q.iloc[0].to_dict(),視窗秒數=w))
        if w in (20,30):
            quality=pd.read_csv(folder/'視窗品質.csv')
            if not quality[meta].sort_values('trip').reset_index(drop=True).equals(ref): raise ValueError(f'W{w}: quality metadata mismatch')
            counts=D.groupby('trip').size()
            if not quality.valid_windows.eq(quality.trip.map(counts).fillna(0)).all(): raise ValueError(f'W{w}: quality counts mismatch')
        cohorts[w]=set(actual.trip)
        distributions.append(D.groupby(['position','cls']).agg(趟數=('trip','nunique'),視窗數=('trip','size')).reset_index().assign(視窗秒數=w))
        for key in ('RF','SVM'):
            name,method=core.MODELS[key]
            path=folder/f'{key}_predictions.csv'
            if w<=15:
                old=pd.read_csv(extended/f'W{w}'/f'{key}_predictions.csv')
                if not old[['row_id']+meta].equals(D[['row_id']+meta]) or not old.pred.isin(ev.LABELS).all():
                    raise ValueError(f'Invalid reused W{w} {key} predictions')
                core.save(old,path)
            elif not path.exists() and (args.window not in (None,w) or args.model not in (None,key)): continue
            model=sc.build_model(ev,name,method)
            T=core.evaluate(D,cols,model,key,ev,'E1',path)
            if not T[['row_id']+meta].equals(D[['row_id']+meta]): raise ValueError('Prediction metadata mismatch')
            predictions[w,key]=T
            V=core.trips(T); core.save(V,folder/f'{key}_每趟結果.csv')
            rows.append(dict(core.summarize(T,ev,key,'E1','all'),視窗秒數=w,分類器=name,方法=sc.METHODS[method],來源='Phase07a' if w<=15 else 'New LORO'))
            ev.plot_cm(T.cls.to_numpy(),T.pred.to_numpy(),f'W{w} {key}',folder/f'{key}_混淆矩陣.png')
            from sklearn.metrics import confusion_matrix
            for level,y,p in [('window',T.cls,T.pred),('trip',V.cls,V.pred)]:
                cm=confusion_matrix(y,p,labels=ev.LABELS)
                for i,a in enumerate(ev.LABELS):
                    for j,b in enumerate(ev.LABELS): cms.append(dict(視窗秒數=w,模型=key,層級=level,實際=a,預測=b,數量=cm[i,j]))
    R=pd.DataFrame(rows); core.save(R,out/'結果_最終視窗長度比較.csv'); core.save(pd.concat(distributions,ignore_index=True),out/'資料分布.csv'); core.save(pd.DataFrame(cms),out/'混淆矩陣數值.csv')
    Q=R.merge(R[R.視窗秒數==15][['模型']+core.METRICS],on='模型',suffixes=('','_W15'),validate='many_to_one')
    for m in core.METRICS: Q[m+'_差異百分點']=(Q[m]-Q[m+'_W15'])*100
    core.save(Q,out/'與15秒比較.csv')
    for m in core.METRICS[:4]: core.save(R.sort_values(m,ascending=False),out/f'排名_{m}.csv')
    core.save(pd.DataFrame(missing_rows,columns=meta+['total_windows','valid_windows','valid_pct','status','視窗秒數']),out/'缺失錄製紀錄.csv')
    # Only publish final common-cohort results once all seven feature cohorts exist.
    if set(cohorts)!=set((1,2,5,10,15,20,30)):
        print('Partial run: common test analysis deferred until all seven feature sets exist.')
        return
    common=set.intersection(*cohorts.values())
    if not common: raise ValueError('Empty common test cohort')
    core.save(ref.assign(共同測試樣本=ref.trip.isin(common)),out/'共同測試樣本名單.csv')
    if set(ref.loc[ref.trip.isin(common),'cls'])!=set(ev.LABELS): raise ValueError('Common cohort lacks a class')
    sensitivity=[]
    for (w,key),T in predictions.items():
        U=T[T.trip.isin(common)]
        if set(U.trip)!=common: raise ValueError('Common test cohort mismatch')
        name,method=core.MODELS[key]
        sensitivity.append(dict(core.summarize(U,ev,key,'E1','all'),視窗秒數=w,分類器=name,方法=sc.METHODS[method],原始訓練資料趟數=T.trip.nunique(),分析方式='共同測試樣本；未重新訓練'))
    S=pd.DataFrame(sensitivity)
    core.save(S,out/'結果_共同測試樣本比較.csv')
    C=S.merge(S[S.視窗秒數==15][['模型']+core.METRICS],on='模型',suffixes=('','_W15'),validate='many_to_one')
    for m in core.METRICS: C[m+'_差異百分點']=(C[m]-C[m+'_W15'])*100
    core.save(C,out/'共同測試樣本_與15秒比較.csv')
    print(R.round(4).to_string(index=False)); print(f'Completed {len(R)}/14 combinations: {out}')
if __name__=='__main__': main()
